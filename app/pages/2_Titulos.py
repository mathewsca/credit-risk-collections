"""Analítico: lista de títulos com filtros (reconstrução da 'Página 2' do relatório original)."""

import streamlit as st
from lib import brl, header, load, require

header("Títulos · Analítico", "Consulta detalhada com filtros e exportação")

t = load("titulos")
if not require(t, "títulos"):
    st.stop()

c = st.columns(5)
fil = c[0].multiselect("Filial", sorted(t.nome_filial.unique()), placeholder="Todas")
rep = c[1].multiselect("Representante", sorted(t.nome_representante.unique()), placeholder="Todos")
esp = c[2].multiselect("Tipo", sorted(t.tipo_atual.unique()), placeholder="Todos")
sta = c[3].multiselect("Aberto/Quitado", ["A", "Q"], placeholder="Ambos")
so_atraso = c[4].toggle("Somente em atraso", value=True)

f = t
for col, sel in (("nome_filial", fil), ("nome_representante", rep), ("tipo_atual", esp), ("status", sta)):
    if sel:
        f = f[f[col].isin(sel)]
if so_atraso:
    f = f[f.vlr_atraso > 0]

st.caption(f"{len(f):,} títulos · valor em atraso {brl(f.vlr_atraso.sum())}".replace(",", "."))
cols = [
    "numero_titulo",
    "numero_parcela",
    "nome_filial",
    "tipo_atual",
    "status",
    "nome_representante",
    "cod_cliente",
    "porte",
    "dt_vencimento",
    "dt_quitacao",
    "dias_atraso",
    "vlr_nominal",
    "vlr_atraso",
    "faixa_aging",
]
view = f.sort_values("vlr_atraso", ascending=False)[cols].head(5000)
st.dataframe(
    view,
    hide_index=True,
    use_container_width=True,
    height=560,
    column_config={
        "numero_titulo": "Título",
        "numero_parcela": "Parcela",
        "nome_filial": "Filial",
        "tipo_atual": "Tipo",
        "status": "A/Q",
        "nome_representante": "Representante",
        "cod_cliente": "Cliente",
        "porte": "Porte",
        "dt_vencimento": st.column_config.DateColumn("Vencimento", format="DD/MM/YYYY"),
        "dt_quitacao": st.column_config.DateColumn("Quitação", format="DD/MM/YYYY"),
        "dias_atraso": "Dias atraso",
        "vlr_nominal": st.column_config.NumberColumn("Nominal", format="R$ %.2f"),
        "vlr_atraso": st.column_config.NumberColumn("Em atraso", format="R$ %.2f"),
        "faixa_aging": "Aging",
    },
)
st.download_button(
    "Baixar CSV",
    view.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
    "titulos.csv",
    "text/csv",
)
