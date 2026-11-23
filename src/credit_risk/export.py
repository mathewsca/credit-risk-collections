"""Exporta um snapshot (Parquet) dos marts para o app público no Streamlit Community Cloud.

Localmente o app lê direto do PostgreSQL; na nuvem gratuita não há banco, então o app usa este snapshot.
"""

from __future__ import annotations

import shutil

from sqlalchemy import inspect

from credit_risk.db import get_engine, read_sql
from credit_risk.settings import ROOT

OUT = ROOT / "app" / "data"

QUERIES = {
    "mensal": "select * from marts.mart_inadimplencia_mensal",
    "aging": "select * from marts.mart_aging",
    "produtos": "select * from marts.mart_produtos_atraso order by qtd_titulos_atraso desc limit 300",
    "empresas": "select * from marts.dim_empresa",
    "titulos": """
        select f.cod_titulo, f.numero_titulo, f.numero_parcela, f.cod_empresa, e.nome_filial, e.uf as uf_filial,
               e.regiao, f.cod_cliente, c.porte, c.uf as uf_cliente, f.cod_representante,
               r.nome_representante, f.tipo_atual, f.status, f.dt_emissao, f.dt_vencimento, f.dt_quitacao,
               f.vlr_nominal::float, f.vlr_pago::float, f.vlr_aberto::float, f.vlr_atraso::float,
               f.dias_atraso, f.faixa_aging, f.is_late30, f.label_disponivel
        from marts.fct_titulos f
        join marts.dim_empresa e using (cod_empresa)
        join marts.dim_cliente c using (cod_cliente)
        join marts.dim_representante r using (cod_representante)
    """,
}
OPTIONAL = {
    "scores": "select * from ml.scores",
    "monitoring": "select * from ml.monitoring",
    "decisions": "select * from ml.model_decisions",
}


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    insp = inspect(get_engine())
    for name, q in QUERIES.items():
        df = read_sql(q)
        df.to_parquet(OUT / f"{name}.parquet", index=False)
        print(f"[export] {name:<10} {len(df):>8,} linhas")
    for name, q in OPTIONAL.items():
        table = q.split("from ")[1].split()[0]
        schema, tbl = table.split(".")
        if insp.has_table(tbl, schema=schema):
            df = read_sql(q)
            df.to_parquet(OUT / f"{name}.parquet", index=False)
            print(f"[export] {name:<10} {len(df):>8,} linhas")
    metrics = ROOT / "reports" / "metrics.json"
    if metrics.exists():
        shutil.copy(metrics, OUT / "metrics.json")
    figs = ROOT / "reports" / "figures"
    if figs.exists():
        shutil.copytree(figs, OUT / "figures", dirs_exist_ok=True)


if __name__ == "__main__":
    run()
