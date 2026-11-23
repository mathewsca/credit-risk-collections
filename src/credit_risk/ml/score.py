"""Scoring em lote da carteira em aberto + fila de priorização da cobrança.

Perda esperada (EL) = PD × saldo em aberto (LGD = 100%, premissa conservadora e explícita).
A fila ordena por EL: com capacidade limitada de cobradores, isso maximiza o valor recuperável
por contato, a mesma lógica de priorização de Pareto usada em Engenharia de Produção.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from credit_risk.ml.features import load_features
from credit_risk.ml.registry import get_alias_version, load_model, predict_pd

BANDS = [(0.01, "A"), (0.03, "B"), (0.10, "C"), (0.30, "D"), (1.01, "E")]


def risk_band(pd_values: np.ndarray) -> np.ndarray:
    out = np.empty(len(pd_values), dtype=object)
    lower = -np.inf
    for upper, label in BANDS:
        out[(pd_values >= lower) & (pd_values < upper)] = label
        lower = upper
    return out


def score_portfolio(df: pd.DataFrame, model) -> pd.DataFrame:
    """Títulos em aberto que ainda NÃO estouraram 30 dias de atraso (onde a ação preventiva ainda funciona)."""
    alvo = df[(df.status == "A") & (~df.is_late30)].copy()
    alvo["pd"] = predict_pd(model, alvo)
    alvo["perda_esperada"] = alvo["pd"] * alvo["vlr_aberto"]
    alvo["faixa_risco"] = risk_band(alvo["pd"].to_numpy())
    alvo = alvo.sort_values("perda_esperada", ascending=False)
    alvo["prioridade"] = np.arange(1, len(alvo) + 1)
    return alvo


def run(features: pd.DataFrame | None = None) -> pd.DataFrame:
    from credit_risk.db import copy_df

    mv = get_alias_version("champion")
    if mv is None:
        raise RuntimeError("Nenhum champion em produção.")
    scored = score_portfolio(load_features(features), load_model("champion"))
    scored["model_version"] = int(mv.version)
    scored["scored_at"] = datetime.now(UTC).replace(tzinfo=None)
    cols = [
        "cod_titulo",
        "cod_cliente",
        "cod_empresa",
        "dt_vencimento",
        "vlr_aberto",
        "pd",
        "perda_esperada",
        "faixa_risco",
        "prioridade",
        "porte",
        "uf_cliente",
        "model_version",
        "scored_at",
    ]
    copy_df(scored[cols], "scores", schema="ml", mode="replace")
    hist = (
        scored.groupby("faixa_risco")
        .agg(
            qtd=("cod_titulo", "size"), saldo=("vlr_aberto", "sum"), perda_esperada=("perda_esperada", "sum")
        )
        .reset_index()
    )
    hist["model_version"], hist["scored_at"] = int(mv.version), scored.scored_at.iloc[0]
    copy_df(hist, "score_history", schema="ml", mode="append")
    print(
        f"[score] {len(scored):,} títulos pontuados com v{mv.version}; "
        f"perda esperada total R$ {scored.perda_esperada.sum():,.0f}"
    )
    return scored


if __name__ == "__main__":
    run()
