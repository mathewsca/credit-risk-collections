"""Carteira: aging, representantes, produtos e concentração (Pareto) da inadimplência."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from lib import SERIES, brl, header, load, pct, require

header("Carteira & Representantes", "Onde está o valor em atraso, e quão concentrado ele é")

t = load("titulos")
if not require(t, "títulos"):
    st.stop()

filiais = st.multiselect("Filial", sorted(t.nome_filial.unique()), placeholder="Todas")
f = t[t.nome_filial.isin(filiais)] if filiais else t

# ---------------- aging
ordem = ["A vencer", "01-30 dias", "31-60 dias", "61-90 dias", "91-180 dias", "180+ dias"]
ag = f[f.status == "A"].groupby("faixa_aging").vlr_aberto.sum().reindex(ordem, fill_value=0).reset_index()
c1, c2 = st.columns(2)
fig = px.bar(
    ag,
    x="faixa_aging",
    y="vlr_aberto",
    title="Saldo em aberto por faixa de atraso (aging)",
    labels={"faixa_aging": "", "vlr_aberto": ""},
    color_discrete_sequence=[SERIES[0]],
)
fig.update_traces(hovertemplate="%{x}<br>R$ %{y:,.0f}<extra></extra>")
fig.update_yaxes(tickprefix="R$ ", tickformat="~s")
c1.plotly_chart(fig, use_container_width=True)

# ---------------- Pareto de clientes (concentração)
cli = f.groupby("cod_cliente").vlr_atraso.sum().sort_values(ascending=False)
cli = cli[cli > 0]
if len(cli):
    cum = cli.cumsum() / cli.sum()
    x = np.arange(1, len(cum) + 1) / len(cum)
    top20 = float(cum.iloc[max(int(len(cum) * 0.2) - 1, 0)])
    fig = go.Figure(
        go.Scatter(
            x=x * 100,
            y=cum.to_numpy() * 100,
            mode="lines",
            line=dict(width=2, color=SERIES[0]),
            hovertemplate="%{x:.1f}% dos clientes<br>%{y:.1f}% do valor em atraso<extra></extra>",
        )
    )
    fig.add_vline(x=20, line_dash="dot", line_color="rgba(128,128,128,.6)")
    fig.add_annotation(
        x=20, y=top20 * 100, text=f"20% dos clientes = {top20:.0%} do atraso", showarrow=True, ax=60, ay=30
    )
    fig.update_layout(
        title="Concentração do atraso (curva de Pareto)",
        xaxis_title="% dos clientes inadimplentes",
        yaxis_title="% acumulado do valor em atraso",
    )
    c2.plotly_chart(fig, use_container_width=True)
    c2.caption("Princípio de Pareto: a cobrança deve começar pela minoria de clientes que concentra o valor.")

# ---------------- representantes
st.markdown("#### Índice por representante")
rep = (
    f.groupby("nome_representante")
    .agg(vlr_nominal=("vlr_nominal", "sum"), vlr_atraso=("vlr_atraso", "sum"), titulos=("cod_titulo", "size"))
    .assign(indice=lambda d: d.vlr_atraso / d.vlr_nominal)
    .sort_values("vlr_atraso", ascending=False)
    .reset_index()
)
st.dataframe(
    rep.head(30),
    hide_index=True,
    use_container_width=True,
    column_config={
        "nome_representante": "Representante",
        "vlr_nominal": st.column_config.NumberColumn("Valor nominal", format="R$ %.0f"),
        "vlr_atraso": st.column_config.NumberColumn("Valor em atraso", format="R$ %.0f"),
        "titulos": "Títulos",
        "indice": st.column_config.ProgressColumn(
            "Índice", format="percent", min_value=0, max_value=float(max(rep.indice.max(), 0.01))
        ),
    },
)

# ---------------- produtos
st.markdown("#### Produtos mais presentes em títulos vencidos")
prod = load("produtos")
if not prod.empty:
    st.dataframe(
        prod.head(100),
        hide_index=True,
        use_container_width=True,
        column_config={
            "cod_produto": "Código",
            "linha": "Linha",
            "qtd_titulos_atraso": "Títulos em atraso",
            "vlr_atraso_titulos": st.column_config.NumberColumn(
                "Valor em atraso dos títulos", format="R$ %.0f"
            ),
        },
    )
    st.caption(
        "Leitura correta: correlação, não causa. Produtos de alto giro aparecem mais em todas as notas."
    )

st.metric(
    "Valor total em atraso (seleção)",
    brl(f.vlr_atraso.sum()),
    help=f"Índice de inadimplência: {pct(f.vlr_atraso.sum() / max(f.vlr_nominal.sum(), 1))}",
)
