"""Camada de dados e tema visual compartilhados pelas páginas do app."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

DATA = Path(__file__).parent / "data"
APP_NAME = "Crédito e Cobrança"

# Padrão visual a partir da paleta "Stock market" (Octet): #9e6976 #992b2b #153e70 #471c24 #0b1b38 #061313.
# Validado para daltonismo (protanopia/deuteranopia) e contraste. Tema da interface em .streamlit/config.toml.
#   séries: marinho da paleta no passo de série (o #153e70 original é escuro demais para série) + tijolo
#   faixas de risco A→E: rampa de um tom, do rosé (risco baixo) ao vinho (risco alto)
#   status: verde/âmbar/vermelho reservados para estável/atenção/mudança, sempre com ícone
SERIES = ["#1f5596", "#992b2b"]
NEUTRO = "#8a8984"  # linha de referência (ex.: ordem por vencimento)
RISK_RAMP = {"A": "#d4a6af", "B": "#bc8894", "C": "#9e6976", "D": "#992b2b", "E": "#471c24"}
STATUS = {
    "estável": ("#0ca30c", "✅"),
    "atenção": ("#fab219", "⚠️"),
    "mudança relevante": ("#d03b3b", "⛔"),
    "sazonal (não dispara)": (NEUTRO, "🗓️"),
}
MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

pio.templates["credito"] = go.layout.Template(
    layout=go.Layout(
        colorway=SERIES,
        font=dict(family="Inter, system-ui, sans-serif", size=13, color="#061313"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=48, b=8),
        xaxis=dict(showgrid=False, linecolor="rgba(128,128,128,.35)", ticks=""),
        yaxis=dict(gridcolor="rgba(128,128,128,.15)", zeroline=False),
        hoverlabel=dict(font_size=12),
        legend=dict(orientation="h", y=1.08, x=0),
        bargap=0.25,
    )
)
pio.templates.default = "plotly+credito"


def _from_postgres(name: str) -> pd.DataFrame | None:
    if not os.getenv("WAREHOUSE_URL"):
        return None
    try:
        from credit_risk.db import read_sql
        from credit_risk.export import OPTIONAL, QUERIES

        return read_sql({**QUERIES, **OPTIONAL}[name])
    except Exception:  # noqa: BLE001 - cai para o snapshot
        return None


@st.cache_data(ttl=600, show_spinner=False)
def load(name: str) -> pd.DataFrame:
    df = _from_postgres(name)
    if df is None:
        f = DATA / f"{name}.parquet"
        df = pd.read_parquet(f) if f.exists() else pd.DataFrame()
    for c in df.columns:
        if c.startswith("dt_") or c in ("mes_vencimento", "run_at", "scored_at", "decided_at"):
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def metrics() -> dict:
    f = DATA / "metrics.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def source_label() -> str:
    return "PostgreSQL (ao vivo)" if os.getenv("WAREHOUSE_URL") else "snapshot Parquet (demo pública)"


def brl(v: float, compact: bool = True) -> str:
    if compact and abs(v) >= 1e6:
        return f"R$ {v / 1e6:,.1f} mi".replace(",", "X").replace(".", ",").replace("X", ".")
    if compact and abs(v) >= 1e3:
        return f"R$ {v / 1e3:,.1f} mil".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(v: float, d: int = 2) -> str:
    return f"{v * 100:.{d}f}%".replace(".", ",")


def header(title: str, subtitle: str) -> None:
    st.set_page_config(page_title=f"{title} · {APP_NAME}", page_icon="📉", layout="wide")
    st.markdown(f"### {title}")
    st.caption(f"{subtitle} · Fonte: {source_label()} · Dados 100% sintéticos")


def require(df: pd.DataFrame, what: str) -> bool:
    if df.empty:
        st.info(f"Sem dados de {what}. Rode `make pipeline` para gerar.")
        return False
    return True
