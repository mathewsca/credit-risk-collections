"""Ingestão: landing (Parquet do ERP) -> schema raw do PostgreSQL, com contrato de dados (pandera).

Idempotente: cada execução substitui as tabelas raw (full refresh). Em produção real, a evolução
natural é carga incremental por data de atualização, mas full refresh é mais simples e suficiente
para o volume deste projeto.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from credit_risk.db import copy_df
from credit_risk.settings import settings

TABLES = ["filiais", "representantes", "clientes", "produtos", "notas", "notas_itens", "titulos"]

# Contrato da tabela mais crítica: falha cedo, antes de contaminar o warehouse.
TITULOS_SCHEMA = pa.DataFrameSchema(
    {
        "titulo_id": pa.Column(int, unique=True),
        "cliente_id": pa.Column(int),
        "valor_nominal": pa.Column(float, pa.Check.gt(0)),
        "valor_pago": pa.Column(float, pa.Check.ge(0)),
        "valor_juros": pa.Column(float, pa.Check.ge(0)),
        "situacao": pa.Column(str, pa.Check.isin(["ABERTO", "PAGO"])),
        "tipo_origem": pa.Column(str, pa.Check.isin(["DUP", "BOL", "CHQ"])),
        "dt_vencimento": pa.Column("datetime64[ns]"),
        "dt_pagamento": pa.Column("datetime64[ns]", nullable=True),
    },
    checks=[
        pa.Check(lambda d: (d.valor_pago <= d.valor_nominal + 0.01).all(), error="pago > nominal"),
        pa.Check(lambda d: (d.dt_emissao <= d.dt_vencimento).all(), error="emissão após vencimento"),
    ],
    coerce=True,
)


def run(landing: Path | None = None) -> dict[str, int]:
    landing = Path(landing or settings.landing_dir)
    loaded_at = datetime.now(UTC).replace(tzinfo=None)
    counts = {}
    for name in TABLES:
        df = pd.read_parquet(landing / f"{name}.parquet")
        if name == "titulos":
            df = TITULOS_SCHEMA.validate(df, lazy=True)
        df["_loaded_at"] = loaded_at
        counts[name] = copy_df(df, name, schema="raw")
        print(f"[ingest] raw.{name:<15} {counts[name]:>9,} linhas")
    return counts


if __name__ == "__main__":
    run()
