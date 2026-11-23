"""Treino do modelo de PD com rastreabilidade completa no MLflow.

Protocolo experimental:
1. Separação temporal train < valid < test (out-of-time).
2. Baseline interpretável (regressão logística) vs. LightGBM com pequena busca de hiperparâmetros.
3. Seleção do modelo pelo PR-AUC na VALIDAÇÃO (o teste é usado uma única vez, para reportar).
4. Calibração isotônica na validação (a PD precisa ser uma probabilidade, não só um ranking).
5. Métricas de teste com IC 95% por bootstrap + comparação com o "oráculo" (PD verdadeira do gerador).
6. Registro no Model Registry com alias "challenger"; a promoção é feita por promote.py.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from matplotlib.colors import LinearSegmentedColormap
from mlflow.models import infer_signature
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from credit_risk.ml.features import CATEGORICAL, FEATURES, NUMERIC, TARGET, load_features, temporal_split
from credit_risk.ml.metrics import bootstrap_ci, evaluate, paired_bootstrap_diff
from credit_risk.settings import ROOT, settings

FIG_DIR = ROOT / "reports" / "figures"
# mesmas cores do app (app/lib.py): marinho e tijolo da paleta "Stock market" (Octet), validadas
SERIES = ["#1f5596", "#992b2b"]
NEUTRO = "#8a8984"
SHAP_CMAP = LinearSegmentedColormap.from_list("marinho_tijolo", [SERIES[0], "#d9d4d6", SERIES[1]])
SEED = 42


def _onehot() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50, sparse_output=False)


def log1p_clip(x):
    return np.log1p(np.clip(x, 0, None))


def build_baseline() -> Pipeline:
    num = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(log1p_clip, feature_names_out="one-to-one")),
            ("sc", StandardScaler()),
        ]
    )
    pre = ColumnTransformer([("num", num, NUMERIC), ("cat", _onehot(), CATEGORICAL)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=0.5, max_iter=3000))])


def build_lgbm(**params) -> Pipeline:
    pre = ColumnTransformer(
        [("num", "passthrough", NUMERIC), ("cat", _onehot(), CATEGORICAL)], verbose_feature_names_out=False
    )
    clf = LGBMClassifier(
        n_estimators=params.get("n_estimators", 400),
        learning_rate=params.get("learning_rate", 0.03),
        num_leaves=params.get("num_leaves", 31),
        min_child_samples=params.get("min_child_samples", 100),
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=SEED,
        deterministic=True,  # resultados reprodutíveis entre execuções (exige force_*_wise)
        force_row_wise=True,
        verbose=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


LGBM_GRID = [
    {"num_leaves": 15, "min_child_samples": 200, "n_estimators": 300, "learning_rate": 0.05},
    {"num_leaves": 31, "min_child_samples": 100, "n_estimators": 400, "learning_rate": 0.03},
    {"num_leaves": 63, "min_child_samples": 50, "n_estimators": 600, "learning_rate": 0.02},
]


TRUSTED_MODULE_PREFIXES = ("credit_risk.", "sklearn.", "lightgbm.", "numpy.", "collections.")


def trusted_types(model) -> list[str]:
    """Allowlist explícita para a serialização segura (skops) do MLflow.

    O skops recusa carregar tipos não declarados (evita execução arbitrária de código, risco do pickle).
    Só liberamos tipos de módulos conhecidos; qualquer tipo inesperado interrompe o registro do modelo.
    """
    import skops.io as sio

    untrusted = sio.get_untrusted_types(data=sio.dumps(model))
    unexpected = [t for t in untrusted if not t.startswith(TRUSTED_MODULE_PREFIXES)]
    if unexpected:
        raise ValueError(f"Tipos não permitidos na serialização do modelo: {unexpected}")
    return untrusted


def calibrate(model: Pipeline, X: pd.DataFrame, y: np.ndarray) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(FrozenEstimator(model), method="isotonic").fit(X, y)


# ----------------------------------------------------------------------------- gráficos
def plot_calibration(y, preds: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 0.3], [0, 0.3], ls="--", c=NEUTRO, lw=1, label="Calibração perfeita")
    for (name, p), cor in zip(preds.items(), SERIES, strict=False):
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", lw=1.5, c=cor, label=name)
    ax.set(
        xlabel="PD prevista (média do decil)",
        ylabel="Taxa observada de atraso > 30d",
        title="Curva de calibração (teste out-of-time)",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_capture(y, p, exposure, path: Path) -> dict[str, float]:
    """Curva de captura (Pareto): % da perda capturada ao priorizar os títulos de maior perda esperada."""
    el = p * exposure
    order = np.argsort(-el)
    loss = (y * exposure)[order]
    cum = np.cumsum(loss) / loss.sum()
    x = np.arange(1, len(cum) + 1) / len(cum)
    rnd = np.cumsum((y * exposure)[np.random.default_rng(0).permutation(len(y))]) / loss.sum()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(x * 100, cum * 100, lw=2, c=SERIES[0], label="Priorização por perda esperada (PD × saldo)")
    ax.plot(x * 100, rnd * 100, ls="--", c=NEUTRO, label="Ordem aleatória")
    for k in (0.1, 0.2):
        ax.axvline(k * 100, c="#ccc", lw=0.8)
    ax.set(
        xlabel="% dos títulos acionados pela cobrança",
        ylabel="% do valor inadimplente capturado",
        title="Curva de captura da cobrança (teste)",
    )
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {f"captura_top{int(k * 100)}": float(cum[int(len(cum) * k) - 1]) for k in (0.05, 0.1, 0.2)}


def plot_shap(model: Pipeline, X: pd.DataFrame, path: Path) -> list[tuple[str, float]]:
    import shap

    pre, clf = model.named_steps["pre"], model.named_steps["clf"]
    Xt = pd.DataFrame(pre.transform(X), columns=pre.get_feature_names_out())
    if isinstance(clf, LogisticRegression):
        # modelo linear: SHAP exato = coeficiente x (valor - média de referência)
        sv = shap.LinearExplainer(clf, Xt).shap_values(Xt)
    else:
        sv = shap.TreeExplainer(clf).shap_values(Xt)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.Series(np.abs(sv).mean(0), index=Xt.columns).sort_values(ascending=False)
    plt.figure()
    shap.summary_plot(sv, Xt, max_display=12, show=False, cmap=SHAP_CMAP)
    plt.title(f"SHAP do modelo selecionado ({type(clf).__name__}), amostra de teste")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return [(k, round(float(v), 4)) for k, v in imp.head(12).items()]


# ----------------------------------------------------------------------------- oráculo
def oracle_metrics(test: pd.DataFrame) -> dict[str, float] | None:
    """PD verdadeira do gerador = limite superior teórico (classificador Bayes-ótimo)."""
    f = settings.truth_dir / "titulos_pd_true.parquet"
    if not f.exists():
        return None
    truth = pd.read_parquet(f)
    m = test[["cod_titulo", TARGET]].merge(truth, left_on="cod_titulo", right_on="titulo_id")
    return evaluate(m[TARGET].astype(int).to_numpy(), m.pd_true.to_numpy())


# ----------------------------------------------------------------------------- main
def run(features: pd.DataFrame | None = None, log_to_mlflow: bool = True) -> dict:
    df = load_features(features)
    split = temporal_split(df)
    Xtr, ytr = split.train[FEATURES], split.train[TARGET].astype(int).to_numpy()
    Xva, yva = split.valid[FEATURES], split.valid[TARGET].astype(int).to_numpy()
    Xte, yte = split.test[FEATURES], split.test[TARGET].astype(int).to_numpy()
    print("[train] split:", json.dumps(split.describe(), ensure_ascii=False))

    # 1) baseline
    base = build_baseline().fit(Xtr, ytr)
    cand = {"logistic": (base, evaluate(yva, base.predict_proba(Xva)[:, 1]), {})}
    # 2) LightGBM: busca pequena, seleção na validação
    for i, params in enumerate(LGBM_GRID):
        m = build_lgbm(**params).fit(Xtr, ytr)
        cand[f"lgbm_{i}"] = (m, evaluate(yva, m.predict_proba(Xva)[:, 1]), params)
    for k, (_, met, _) in cand.items():
        print(f"[train] valid {k:<9} pr_auc={met['pr_auc']:.4f} roc_auc={met['roc_auc']:.4f}")
    best_lgbm = max((k for k in cand if k.startswith("lgbm")), key=lambda k: cand[k][1]["pr_auc"])
    # Regra de parcimônia: o modelo complexo só é escolhido se superar o baseline com significância
    # (limite inferior do IC 95% da diferença pareada de PR-AUC > 0 na VALIDAÇÃO).
    selection = paired_bootstrap_diff(
        yva, cand[best_lgbm][0].predict_proba(Xva)[:, 1], cand["logistic"][0].predict_proba(Xva)[:, 1]
    )
    selected = best_lgbm if selection["ci_low"] > 0 else "logistic"
    print(
        f"[train] PR-AUC LightGBM - logística (validação) = {selection['diff']:+.4f} "
        f"IC95% [{selection['ci_low']:+.4f}; {selection['ci_high']:+.4f}] -> {selected}"
    )

    # 3) calibração na validação + avaliação única no teste
    final = {k: calibrate(cand[k][0], Xva, yva) for k in ("logistic", best_lgbm)}
    p_test = {k: m.predict_proba(Xte)[:, 1] for k, m in final.items()}
    results = {k: {"metrics": evaluate(yte, p), "ci95": bootstrap_ci(yte, p)} for k, p in p_test.items()}
    oracle = oracle_metrics(split.test)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_calibration(
        yte,
        {"Regressão logística": p_test["logistic"], "LightGBM": p_test[best_lgbm]},
        FIG_DIR / "calibracao.png",
    )
    capture = plot_capture(yte, p_test[selected], split.test.vlr_nominal.to_numpy(), FIG_DIR / "captura.png")
    shap_top = plot_shap(
        cand[selected][0], Xte.sample(min(3000, len(Xte)), random_state=SEED), FIG_DIR / "shap.png"
    )

    summary = {
        "split": split.describe(),
        "selected": "lightgbm" if selected.startswith("lgbm") else "logistic",
        "lgbm_params": cand[best_lgbm][2],
        "selection_valid": selection,
        "test": {("lightgbm" if k.startswith("lgbm") else k): v for k, v in results.items()},
        "oracle": oracle,
        "capture": capture,
        "shap_top": shap_top,
    }
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    sel = summary["test"][summary["selected"]]["metrics"]
    print(
        f"[train] selecionado={summary['selected']} | teste pr_auc={sel['pr_auc']:.4f} "
        f"roc_auc={sel['roc_auc']:.4f} brier={sel['brier']:.5f}"
    )
    if oracle:
        print(f"[train] oráculo pr_auc={oracle['pr_auc']:.4f} roc_auc={oracle['roc_auc']:.4f}")

    if log_to_mlflow:
        # referência de drift = população mais recente no momento do treino (validação + teste)
        reference = pd.concat([Xva, Xte])
        summary["model_version"] = _log_mlflow(final[selected], summary, Xtr, reference)
    return summary


def _log_mlflow(model, summary: dict, Xtr: pd.DataFrame, reference: pd.DataFrame) -> str:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)
    sel = summary["selected"]
    with mlflow.start_run(run_name=f"train-{sel}") as run:
        mlflow.log_params(
            {
                "selected_model": sel,
                "features": ",".join(FEATURES),
                "calibration": "isotonic",
                **{f"split_{k}": v for k, v in summary["split"].items()},
                **{f"lgbm_{k}": v for k, v in summary["lgbm_params"].items()},
            }
        )
        for algo, res in summary["test"].items():
            mlflow.log_metrics({f"test_{algo}_{k}": v for k, v in res["metrics"].items()})
            mlflow.log_metrics({f"test_{algo}_{k}_ci_low": v[0] for k, v in res["ci95"].items()})
            mlflow.log_metrics({f"test_{algo}_{k}_ci_high": v[1] for k, v in res["ci95"].items()})
        if summary["oracle"]:
            mlflow.log_metrics({f"oracle_{k}": v for k, v in summary["oracle"].items()})
        mlflow.log_metrics(summary["capture"])
        mlflow.log_artifacts(str(FIG_DIR), artifact_path="figures")
        mlflow.log_dict(summary, "metrics_summary.json")

        # Referência para monitoramento de drift (features + distribuição de score)
        with tempfile.TemporaryDirectory() as tmp:
            ref = reference.sample(min(20000, len(reference)), random_state=SEED).copy()
            ref["pd"] = model.predict_proba(ref[FEATURES])[:, 1]
            ref.to_parquet(Path(tmp) / "reference.parquet", index=False)
            mlflow.log_artifact(str(Path(tmp) / "reference.parquet"), artifact_path="monitoring")

        example = Xtr.head(5)
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=settings.model_name,
            signature=infer_signature(example, model.predict_proba(example)[:, 1]),
            input_example=example,
            serialization_format="skops",
            skops_trusted_types=trusted_types(model),
        )
        version = str(info.registered_model_version)
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(settings.model_name, "challenger", version)
        m = summary["test"][sel]["metrics"]
        for k in ("pr_auc", "roc_auc", "brier", "ks"):
            client.set_model_version_tag(settings.model_name, version, f"test_{k}", f"{m[k]:.6f}")
        client.set_model_version_tag(settings.model_name, version, "train_run_id", run.info.run_id)
        print(f"[train] registrado {settings.model_name} v{version} (alias=challenger)")
        return version


if __name__ == "__main__":
    run()
