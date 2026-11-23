"""Métricas de discriminação, calibração e estabilidade com incerteza quantificada."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    """Kolmogorov-Smirnov: máxima distância entre as CDFs de score de bons e maus (padrão em crédito)."""
    y, p = np.asarray(y).astype(bool), np.asarray(p)
    order = np.argsort(p)
    y_sorted = y[order]
    cdf_bad = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    cdf_good = np.cumsum(~y_sorted) / max((~y_sorted).sum(), 1)
    return float(np.max(np.abs(cdf_bad - cdf_good)))


METRICS = {
    "roc_auc": roc_auc_score,
    "pr_auc": average_precision_score,
    "brier": brier_score_loss,
    "ks": ks_statistic,
}


def evaluate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    out = {name: float(fn(y, p)) for name, fn in METRICS.items()}
    out["taxa_observada"] = float(np.mean(y))
    out["pd_media"] = float(np.mean(p))
    return out


def bootstrap_ci(
    y: np.ndarray, p: np.ndarray, n_boot: int = 500, alpha: float = 0.05, seed: int = 0
) -> dict[str, tuple[float, float]]:
    """IC percentil por bootstrap não paramétrico (Efron & Tibshirani, 1993)."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    samples: dict[str, list[float]] = {k: [] for k in METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        for name, fn in METRICS.items():
            samples[name].append(fn(y[idx], p[idx]))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for k, v in samples.items()}


def paired_bootstrap_diff(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    metric=average_precision_score,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Diferença metric(A) - metric(B) com IC por bootstrap PAREADO (mesmas reamostragens para A e B).

    Parear remove a variância comum às duas avaliações: é o teste adequado para decidir se um
    modelo mais complexo realmente supera o mais simples nos MESMOS dados.
    """
    rng = np.random.default_rng(seed)
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(metric(y[idx], p_a[idx]) - metric(y[idx], p_b[idx]))
    diffs = np.asarray(diffs)
    return {
        "diff": float(metric(y, p_a) - metric(y, p_b)),
        "ci_low": float(np.percentile(diffs, 100 * alpha / 2)),
        "ci_high": float(np.percentile(diffs, 100 * (1 - alpha / 2))),
        "p_a_melhor": float((diffs > 0).mean()),
    }


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index (Siddiqi, 2006).

    Faixas usuais: < 0,10 estável | 0,10-0,25 atenção | > 0,25 mudança relevante.
    Bins por quantis da distribuição de referência; robusto a variáveis com muitos empates.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected, actual = expected[~np.isnan(expected)], actual[~np.isnan(actual)]
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # variável quase constante -> compara proporções por valor
        values = np.union1d(np.unique(expected), np.unique(actual))
        e = np.array([(expected == v).mean() for v in values])
        a = np.array([(actual == v).mean() for v in values])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(expected, actual, eps: float = 1e-4) -> float:
    import pandas as pd

    e = pd.Series(expected).value_counts(normalize=True)
    a = pd.Series(actual).value_counts(normalize=True)
    idx = e.index.union(a.index)
    e = e.reindex(idx, fill_value=0).clip(lower=eps)
    a = a.reindex(idx, fill_value=0).clip(lower=eps)
    return float(((a - e) * np.log(a / e)).sum())


def psi_status(value: float) -> str:
    return "estável" if value < 0.10 else ("atenção" if value < 0.25 else "mudança relevante")
