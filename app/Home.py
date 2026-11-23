"""Visão Geral: reconstrução da página inicial do dashboard de inadimplência."""

import plotly.express as px
import streamlit as st
from lib import MESES, SERIES, brl, header, load, pct, require

header("Inadimplência · Visão Geral", "Contas a receber por filial, UF, porte e mês de vencimento")

t = load("titulos")
if not require(t, "títulos"):
    st.stop()

# ---------------- filtros (uma linha, acima dos gráficos)
c1, c2, c3 = st.columns([3, 1.2, 2])
filiais = c1.multiselect("Filial", sorted(t.nome_filial.unique()), placeholder="Todas")
regioes = c2.multiselect("Região", sorted(t.regiao.unique()), placeholder="Todas")
# Só vencimentos até a data de referência: títulos "a vencer" teriam índice 0% por definição (enganoso)
referencia = t.dt_quitacao.max().to_period("M")
t = t[t.dt_vencimento.dt.to_period("M") <= referencia]
meses = sorted(t.dt_vencimento.dt.to_period("M").unique())
ini, fim = c3.select_slider(
    "Mês de vencimento",
    options=meses,
    value=(meses[0], meses[-1]),
    format_func=lambda p: f"{MESES[p.month - 1]}/{p.year % 100:02d}",
)
f = t[(t.dt_vencimento.dt.to_period("M") >= ini) & (t.dt_vencimento.dt.to_period("M") <= fim)]
if filiais:
    f = f[f.nome_filial.isin(filiais)]
if regioes:
    f = f[f.regiao.isin(regioes)]


def indice(df):
    return df.vlr_atraso.sum() / max(df.vlr_nominal.sum(), 1)


# ---------------- KPIs
k = st.columns(4)
k[0].metric("Valor nominal", brl(f.vlr_nominal.sum()))
k[1].metric("Valor pago", brl(f.vlr_pago.sum()))
k[2].metric("Valor em atraso", brl(f.vlr_atraso.sum()))
k[3].metric(
    "Índice de inadimplência",
    pct(indice(f)),
    help="Σ valor em atraso ÷ Σ valor nominal (medida DAX original)",
)

# ---------------- gráficos
g1, g2 = st.columns([1, 1])
by_f = (
    f.groupby("nome_filial")
    .apply(indice, include_groups=False)
    .rename("indice")
    .reset_index()
    .sort_values("indice")
    .tail(15)
)
fig = px.bar(
    by_f,
    x="indice",
    y="nome_filial",
    orientation="h",
    title="Índice por filial (15 maiores)",
    labels={"indice": "", "nome_filial": ""},
    color_discrete_sequence=[SERIES[0]],
)
fig.update_traces(hovertemplate="%{y}<br>Índice: %{x:.2%}<extra></extra>")
fig.update_xaxes(tickformat=".1%")
g1.plotly_chart(fig, use_container_width=True)

mens = f.assign(mes=f.dt_vencimento.dt.to_period("M").dt.to_timestamp())
mens = mens.groupby("mes").apply(indice, include_groups=False).rename("indice").reset_index()
fig = px.line(
    mens,
    x="mes",
    y="indice",
    title="Índice por mês de vencimento",
    markers=True,
    labels={"mes": "", "indice": ""},
    color_discrete_sequence=[SERIES[0]],
)
fig.update_traces(line_width=2, marker_size=8, hovertemplate="%{x|%b/%Y}<br>Índice: %{y:.2%}<extra></extra>")
fig.update_yaxes(tickformat=".1%", rangemode="tozero")
fig.update_layout(hovermode="x unified")
g2.plotly_chart(fig, use_container_width=True)
g2.caption(
    "Meses recentes têm índice maior porque parte dos títulos ainda está dentro do prazo de "
    "pagamento: o índice é uma foto da data do snapshot, e não uma taxa por safra."
)

g3, g4 = st.columns(2)
uf = (
    f.groupby("uf_filial")
    .apply(indice, include_groups=False)
    .rename("indice")
    .reset_index()
    .sort_values("indice")
)
fig = px.bar(
    uf,
    x="uf_filial",
    y="indice",
    title="Índice por UF da filial",
    labels={"uf_filial": "", "indice": ""},
    color_discrete_sequence=[SERIES[0]],
)
fig.update_traces(hovertemplate="%{x}<br>Índice: %{y:.2%}<extra></extra>")
fig.update_yaxes(tickformat=".1%")
g3.plotly_chart(fig, use_container_width=True)

po = (
    f.groupby("porte").apply(indice, include_groups=False).rename("indice").reset_index().sort_values("porte")
)
fig = px.bar(
    po,
    x="porte",
    y="indice",
    title="Índice por porte do cliente",
    labels={"porte": "", "indice": ""},
    color_discrete_sequence=[SERIES[0]],
)
fig.update_traces(hovertemplate="%{x}<br>Índice: %{y:.2%}<extra></extra>")
fig.update_yaxes(tickformat=".1%")
g4.plotly_chart(fig, use_container_width=True)

with st.expander("Como este painel foi reconstruído"):
    st.markdown(
        "- Relatório de inadimplência originalmente feito em Power BI, reconstruído sobre um **data warehouse "
        "PostgreSQL modelado com dbt**.\n"
        "- O **gráfico de pizza** por porte virou **barras**: comparar ângulos é menos preciso que comparar "
        "comprimentos (Cleveland & McGill, 1984).\n"
        "- As regras de negócio (`ATRASO`, `VLRATRASO`, `Indice Inad`) foram traduzidas de DAX/SQL Server "
        "para SQL versionado e testado (ver `docs/dax_to_sql.md`)."
    )
