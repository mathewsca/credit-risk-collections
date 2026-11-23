import numpy as np
import pandas as pd
import pytest

from credit_risk.ml.features import CATEGORICAL, NUMERIC


@pytest.fixture(scope="session")
def features_df() -> pd.DataFrame:
    """Tabela de features sintética mínima (sem banco) com sinal conhecido em hist_taxa_late30."""
    rng = np.random.default_rng(0)
    n = 6000
    safra = pd.to_datetime(rng.choice(pd.date_range("2023-01-01", "2024-12-01", freq="MS"), n))
    df = pd.DataFrame({c: rng.gamma(2, 2, n) for c in NUMERIC})
    df["hist_taxa_late30"] = rng.beta(0.5, 10, n)
    df["mes_vencimento"] = safra.month
    for c in CATEGORICAL:
        df[c] = rng.choice(["a", "b", "c"], n)
    df["tipo_origem"] = rng.choice(["DUP", "BOL", "CHQ"], n, p=[0.9, 0.05, 0.05])
    df["sem_representante"] = rng.choice([True, False], n)
    logit = -3.5 + 30 * df.hist_taxa_late30 + 0.5 * (df.tipo_origem == "BOL")
    df["is_late30"] = rng.random(n) < 1 / (1 + np.exp(-logit))
    df["cod_titulo"] = np.arange(n)
    df["cod_cliente"] = rng.integers(0, 500, n)
    df["safra_vencimento"] = safra
    df["dt_vencimento"] = safra + pd.Timedelta(days=10)
    df["dt_emissao"] = safra - pd.Timedelta(days=20)
    df["label_disponivel"] = safra < "2024-12-01"
    df["status"] = np.where(df.label_disponivel, "Q", "A")
    df["vlr_aberto"] = np.where(df.status == "A", df.vlr_nominal, 0.0)
    df["porte"], df["uf_cliente"] = "03 - Médio", "GO"
    return df
