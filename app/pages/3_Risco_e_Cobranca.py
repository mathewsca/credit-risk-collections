"""Risco e priorização: transforma a PD do modelo em uma fila de cobrança com capacidade limitada."""

import os

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from lib import NEUTRO, RISK_RAMP, SERIES, brl, header, load, require

header(
    "Risco & Priorização da Cobrança", "PD por título → perda esperada → fila ótima para a equipe de cobrança"
)

s = load("scores")
if not require(s, "scores (rode o pipeline com um modelo champion)"):
    st.stop()

k = st.columns(4)
k[0].metric(
    "Títulos monitorados", f"{len(s):,}".replace(",", "."), help="Em aberto e ainda sem 30 dias de atraso"
)
k[1].metric("Saldo", brl(s.vlr_aberto.sum()))
k[2].metric(
    "Perda esperada", brl(s.perda_esperada.sum()), help="Σ PD × saldo (LGD = 100%, premissa conservadora)"
)
k[3].metric("Modelo em produção", f"v{int(s.model_version.iloc[0])}")

c1, c2 = st.columns(2)
fx = (
    s.groupby("faixa_risco")
    .agg(qtd=("cod_titulo", "size"), perda=("perda_esperada", "sum"))
    .reindex(list(RISK_RAMP), fill_value=0)
    .reset_index()
)
fig = px.bar(
    fx,
    x="faixa_risco",
    y="perda",
    color="faixa_risco",
    color_discrete_map=RISK_RAMP,
    title="Perda esperada por faixa de risco",
    labels={"faixa_risco": "Faixa", "perda": ""},
    custom_data=["qtd"],
)
fig.update_traces(
    hovertemplate="Faixa %{x}<br>Perda esperada: R$ %{y:,.0f}<br>%{customdata[0]} títulos<extra></extra>",
    texttemplate="%{customdata[0]} tít.",
    textposition="outside",
)
fig.update_layout(showlegend=False)
fig.update_yaxes(tickprefix="R$ ", tickformat="~s")
c1.plotly_chart(fig, use_container_width=True)
c1.caption("Faixas: A < 1% · B 1–3% · C 3–10% · D 10–30% · E ≥ 30% de PD.")

# ---------------- simulador de capacidade
cap = c2.slider(
    "Capacidade da equipe (títulos acionados por ciclo)", 50, min(len(s), 5000), min(500, len(s)), step=50
)
el = s.perda_esperada.to_numpy()
by_el = np.cumsum(np.sort(el)[::-1]) / el.sum()
by_value = np.cumsum(el[np.argsort(-s.vlr_aberto.to_numpy())]) / el.sum()
by_due = np.cumsum(el[np.argsort(s.dt_vencimento.to_numpy())]) / el.sum()
x = np.arange(1, len(el) + 1)
fig = go.Figure()
for y, name, color in (
    (by_el, "Por perda esperada (modelo)", SERIES[0]),
    (by_value, "Por maior saldo", SERIES[1]),
    (by_due, "Por vencimento mais próximo", NEUTRO),
):
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y * 100,
            name=name,
            mode="lines",
            line=dict(width=2, color=color),
            hovertemplate="%{x} títulos: %{y:.1f}%<extra>" + name + "</extra>",
        )
    )
fig.add_vline(x=cap, line_dash="dot", line_color="rgba(128,128,128,.7)")
fig.update_layout(
    title="Perda esperada coberta × títulos acionados",
    xaxis_title="Títulos acionados",
    yaxis_title="% da perda esperada coberta",
    hovermode="x unified",
)
c2.plotly_chart(fig, use_container_width=True)
ganho = by_el[cap - 1] - by_due[cap - 1]
c2.success(
    f"Com **{cap}** acionamentos, a fila do modelo cobre **{by_el[cap - 1]:.0%}** da perda esperada, "
    f"contra {by_due[cap - 1]:.0%} na ordem por vencimento (+{ganho * 100:.0f} p.p.)."
)

st.markdown("#### Fila de cobrança priorizada")
st.dataframe(
    s.sort_values("prioridade").head(300),
    hide_index=True,
    use_container_width=True,
    column_config={
        "prioridade": "Prioridade",
        "cod_titulo": "Título",
        "cod_cliente": "Cliente",
        "porte": "Porte",
        "uf_cliente": "UF",
        "dt_vencimento": st.column_config.DateColumn("Vencimento", format="DD/MM/YYYY"),
        "vlr_aberto": st.column_config.NumberColumn("Saldo", format="R$ %.2f"),
        "pd": st.column_config.ProgressColumn("PD", format="percent", min_value=0, max_value=1),
        "perda_esperada": st.column_config.NumberColumn("Perda esperada", format="R$ %.2f"),
        "faixa_risco": "Faixa",
        "cod_empresa": None,
        "model_version": None,
        "scored_at": None,
    },
    column_order=[
        "prioridade",
        "cod_titulo",
        "cod_cliente",
        "porte",
        "uf_cliente",
        "dt_vencimento",
        "vlr_aberto",
        "pd",
        "perda_esperada",
        "faixa_risco",
    ],
)

# ---------------- API ao vivo (somente no ambiente Docker)
api = os.getenv("API_URL")
if api:
    with st.expander("Testar a API de scoring ao vivo (FastAPI)"):
        import requests

        a, b, c = st.columns(3)
        payload = {
            "vlr_nominal": a.number_input("Valor do título (R$)", 10.0, 50000.0, 850.0),
            "numero_parcela": int(b.number_input("Nº da parcela", 1, 12, 2)),
            "qtd_parcelas": int(c.number_input("Qtd. parcelas", 1, 12, 3)),
            "prazo_dias": int(a.number_input("Prazo (dias)", 0, 400, 60)),
            "mes_vencimento": int(b.number_input("Mês de vencimento", 1, 12, 3)),
            "tempo_cliente_anos": c.number_input("Tempo de cliente (anos)", 0.0, 30.0, 4.0),
            "hist_qtd_titulos": int(a.number_input("Títulos anteriores", 0, 2000, 30)),
            "hist_taxa_late30": b.slider("Taxa histórica de atraso > 30d", 0.0, 1.0, 0.02),
            "hist_qtd_late30_180d": int(c.number_input("Atrasos > 30d (últimos 180d)", 0, 200, 0)),
            "tipo_origem": a.selectbox("Tipo do título", ["DUP", "BOL", "CHQ"]),
            "porte": b.selectbox(
                "Porte",
                ["Nao Cadastrado", "01 - Micro", "02 - Pequeno", "03 - Médio", "04 - Grande", "05 - Master"],
                index=3,
            ),
            "uf_cliente": c.text_input("UF do cliente", "GO"),
        }
        if st.button("Calcular PD"):
            try:
                r = requests.post(f"{api}/predict", json=[payload], timeout=10)
                r.raise_for_status()
                st.json(r.json()[0])
            except Exception as exc:  # noqa: BLE001
                st.error(f"API indisponível: {exc}")
