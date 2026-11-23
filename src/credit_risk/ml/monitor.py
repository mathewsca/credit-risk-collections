"""Monitoramento do modelo em produção e gatilho de retreino.

Três sinais complementares (com limiares documentados):
1. Estabilidade da população (PSI das features e do score): população recente vs. referência de treino.
   O rótulo demora 31+ dias para existir; PSI é o alerta antecipado.
2. Desempenho na safra mais recente já maturada: ROC-AUC vs. o AUC de teste do modelo.
3. Calibração-no-agregado: taxa observada / PD média prevista (O/E). Fora de [0,67; 1,5] = descalibrado.

Saída: dicionário com `retrain` (bool) e `reasons`; métricas gravadas em ml.monitoring.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from sklearn.metrics import roc_auc_score

from credit_risk.ml.features import CATEGORICAL, NUMERIC, TARGET, load_features
from credit_risk.ml.metrics import psi, psi_categorical, psi_status
from credit_risk.ml.registry import get_alias_version, load_model, load_reference, predict_pd

PSI_LIMIT = 0.25
AUC_DROP_LIMIT = 0.05
OE_RANGE = (0.67, 1.5)
RECENT_DAYS = 90
# Variáveis de calendário mudam de distribuição POR CONSTRUÇÃO entre janelas de tempo diferentes
# (ex.: jul-dez vs nov-jan). São reportadas, mas não disparam retreino (seria falso alarme).
CALENDAR = {"mes_vencimento"}


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # só compara variáveis presentes nos dois lados (a referência pertence à versão do modelo em produção)
    for c in [c for c in NUMERIC if c in reference]:
        rows.append((c, psi(reference[c].to_numpy(), current[c].to_numpy())))
    for c in [c for c in CATEGORICAL if c in reference]:
        rows.append((c, psi_categorical(reference[c].astype(str), current[c].astype(str))))
    rows.append(("score_pd", psi(reference["pd"].to_numpy(), current["pd"].to_numpy())))
    out = pd.DataFrame(rows, columns=["item", "psi"])
    out["status"] = out.psi.map(psi_status)
    out.loc[out.item.isin(CALENDAR), "status"] = "sazonal (não dispara)"
    return out


def run(features: pd.DataFrame | None = None) -> dict:
    champ = get_alias_version("champion")
    if champ is None:
        return {"retrain": True, "reasons": ["sem modelo champion em produção"]}

    df = load_features(features)
    model = load_model("champion")
    reference = load_reference("champion")
    ref_date = df.dt_emissao.max()

    # (1) população recente: títulos emitidos nos últimos 90 dias
    current = df[df.dt_emissao > ref_date - pd.Timedelta(days=RECENT_DAYS)].copy()
    current["pd"] = predict_pd(model, current)
    drift = compute_drift(reference, current)

    # (2) e (3) safra mais recente com rótulo maturado
    lab = df[df.label_disponivel]
    last = lab[lab.safra_vencimento == lab.safra_vencimento.max()]
    p_last = predict_pd(model, last)
    y_last = last[TARGET].astype(int).to_numpy()
    auc_recent = float(roc_auc_score(y_last, p_last)) if 0 < y_last.sum() < len(y_last) else float("nan")
    auc_ref = float(champ.tags.get("test_roc_auc", "nan"))
    oe = float(y_last.mean() / max(p_last.mean(), 1e-9))

    reasons = []
    score_psi = float(drift.loc[drift.item == "score_pd", "psi"].iloc[0])
    if score_psi > PSI_LIMIT:
        reasons.append(f"PSI do score = {score_psi:.3f} > {PSI_LIMIT}")
    eligible = (drift.item != "score_pd") & ~drift.item.isin(CALENDAR)
    drifted = drift[eligible & (drift.psi > PSI_LIMIT)].item.tolist()
    if drifted:
        reasons.append(f"features com PSI > {PSI_LIMIT}: {', '.join(drifted)}")
    if auc_ref == auc_ref and auc_recent == auc_recent and auc_ref - auc_recent > AUC_DROP_LIMIT:
        reasons.append(f"queda de ROC-AUC {auc_ref:.3f} -> {auc_recent:.3f}")
    if not OE_RANGE[0] <= oe <= OE_RANGE[1]:
        reasons.append(f"descalibração: observado/esperado = {oe:.2f}")

    run_at = datetime.now(UTC).replace(tzinfo=None)
    metrics = pd.concat(
        [
            drift.assign(tipo="psi").rename(columns={"psi": "valor"}),
            pd.DataFrame(
                [
                    {
                        "item": "roc_auc_safra_recente",
                        "valor": auc_recent,
                        "status": "",
                        "tipo": "desempenho",
                    },
                    {"item": "roc_auc_referencia", "valor": auc_ref, "status": "", "tipo": "desempenho"},
                    {"item": "observado_sobre_esperado", "valor": oe, "status": "", "tipo": "calibracao"},
                ]
            ),
        ]
    )
    metrics["run_at"], metrics["model_version"] = run_at, int(champ.version)
    metrics["safra_avaliada"] = last.safra_vencimento.max().date()
    metrics["retrain"] = bool(reasons)
    try:
        from credit_risk.db import copy_df

        copy_df(metrics, "monitoring", schema="ml", mode="append")
    except Exception as exc:  # noqa: BLE001
        print(f"[monitor] aviso: não gravou métricas ({exc})")

    result = {
        "retrain": bool(reasons),
        "reasons": reasons,
        "score_psi": score_psi,
        "auc_recent": auc_recent,
        "auc_ref": auc_ref,
        "observed_expected": oe,
        "model_version": champ.version,
    }
    print(
        f"[monitor] v{champ.version} | PSI score={score_psi:.3f} | AUC recente={auc_recent:.3f} "
        f"(ref {auc_ref:.3f}) | O/E={oe:.2f} | retreinar={result['retrain']} {reasons}"
    )
    return result


if __name__ == "__main__":
    run()
