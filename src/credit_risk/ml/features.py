"""Definição das features, carregamento e separação temporal (out-of-time)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

NUMERIC = [
    "vlr_nominal",
    "numero_parcela",
    "qtd_parcelas",
    "mes_vencimento",
    "tempo_cliente_anos",
    "hist_qtd_titulos",
    "hist_taxa_late30",
    "hist_qtd_late30_180d",
    "hist_max_dias_atraso",
    "exp_qtd_titulos_abertos",
    "exp_vlr_aberto",
    "atual_qtd_vencidos",
    "rec_media_dias_atraso_90d",
    "rec_qtd_pagamentos_90d",
]
CATEGORICAL = ["tipo_origem", "porte", "uf_cliente", "divisao", "sem_representante"]

# Removidas por redundância (|rho de Spearman| > 0,9 com a feature mantida; ver docs/model_card.md).
# Continuam em ml.ml_features para análise, mas fora do modelo: colinearidade torna os coeficientes
# da regressão logística instáveis e com sinais contraditórios, prejudicando a interpretação.
REDUNDANT = {
    "hist_dias_desde_ultimo_late30": "hist_taxa_late30",
    "atual_vlr_vencido": "atual_qtd_vencidos",
    "atual_max_dias_vencido": "atual_qtd_vencidos",
    "prazo_dias": "numero_parcela",
    "uf_filial": "uf_cliente",
}
FEATURES = NUMERIC + CATEGORICAL
TARGET = "is_late30"


def load_features(source: str | pd.DataFrame | None = None) -> pd.DataFrame:
    """Lê ml.ml_features do warehouse (padrão), de um Parquet ou de um DataFrame já carregado."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, str) and source.endswith(".parquet"):
        df = pd.read_parquet(source)
    else:
        from credit_risk.db import read_sql

        df = read_sql("select * from ml.ml_features")
    for c in ("dt_emissao", "dt_vencimento", "safra_vencimento"):
        df[c] = pd.to_datetime(df[c])
    df["sem_representante"] = df["sem_representante"].astype(str)
    return df


@dataclass
class TemporalSplit:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    def describe(self) -> dict[str, str]:
        def rng(d: pd.DataFrame) -> str:
            return f"{d.safra_vencimento.min():%Y-%m} a {d.safra_vencimento.max():%Y-%m} (n={len(d):,})"

        return {"train": rng(self.train), "valid": rng(self.valid), "test": rng(self.test)}


def temporal_split(df: pd.DataFrame, valid_months: int = 3, test_months: int = 3) -> TemporalSplit:
    """Separação por safra de vencimento, apenas com rótulos já maturados.

    train < valid < test no tempo: simula o uso real (treinar no passado, prever o futuro) e evita
    que padrões do futuro vazem para o treino, algo que uma separação aleatória não garante.
    """
    lab = df[df.label_disponivel].copy()
    last = lab.safra_vencimento.max()
    test_start = last - pd.DateOffset(months=test_months - 1)
    valid_start = test_start - pd.DateOffset(months=valid_months)
    return TemporalSplit(
        train=lab[lab.safra_vencimento < valid_start],
        valid=lab[(lab.safra_vencimento >= valid_start) & (lab.safra_vencimento < test_start)],
        test=lab[lab.safra_vencimento >= test_start],
    )
