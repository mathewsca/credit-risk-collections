"""MLOps: evidência de desempenho do modelo, estabilidade (PSI) e trilha de auditoria das promoções."""

from pathlib import Path

import pandas as pd
import streamlit as st
from lib import STATUS, header, load, metrics

header("Monitoramento do Modelo", "Desempenho out-of-time com IC 95%, drift (PSI) e decisões de promoção")

m = metrics()
if m:
    st.markdown("#### Desempenho no teste out-of-time")
    st.caption(
        f"Treino: {m['split']['train']} · Validação: {m['split']['valid']} · Teste: {m['split']['test']}"
    )
    rows = []
    for algo, res in m["test"].items():
        for k in ("roc_auc", "pr_auc", "ks", "brier"):
            lo, hi = res["ci95"][k]
            rows.append(
                {
                    "Modelo": algo + (" ✔ selecionado" if algo == m["selected"] else ""),
                    "Métrica": k,
                    "Valor": f"{res['metrics'][k]:.4f}",
                    "IC 95%": f"[{lo:.4f}; {hi:.4f}]",
                }
            )
    if m.get("oracle"):
        for k in ("roc_auc", "pr_auc", "ks", "brier"):
            rows.append(
                {
                    "Modelo": "oráculo (PD verdadeira)",
                    "Métrica": k,
                    "Valor": f"{m['oracle'][k]:.4f}",
                    "IC 95%": "limite teórico",
                }
            )
    tab = pd.DataFrame(rows).pivot(index="Modelo", columns="Métrica", values="Valor")
    st.dataframe(tab[["roc_auc", "pr_auc", "ks", "brier"]], use_container_width=True)
    with st.expander("Intervalos de confiança (bootstrap, 500 reamostragens)"):
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        f"Taxa de eventos no teste: {m['test'][m['selected']]['metrics']['taxa_observada']:.2%}. "
        "Com classes desbalanceadas, o PR-AUC é mais informativo que o ROC-AUC (Saito & Rehmsmeier, 2015)."
    )

    figs = Path(__file__).parents[1] / "data" / "figures"
    cols = st.columns(3)
    for col, (f, cap) in zip(
        cols,
        [
            ("calibracao.png", "Calibração: PD prevista × observada"),
            ("captura.png", "Curva de captura da cobrança"),
            ("shap.png", "Explicabilidade (SHAP)"),
        ],
        strict=True,
    ):
        if (figs / f).exists():
            col.image(str(figs / f), caption=cap, use_container_width=True)

st.markdown("#### Estabilidade da população (PSI) na última execução")
mon = load("monitoring")
if mon.empty:
    st.info("Sem execuções de monitoramento ainda.")
else:
    last = mon[mon.run_at == mon.run_at.max()]
    psi = last[last.tipo == "psi"].sort_values("valor", ascending=False)
    psi["Status"] = psi.status.map(lambda s: f"{STATUS.get(s, ('', ''))[1]} {s}")
    c1, c2 = st.columns([2, 1])
    c1.dataframe(
        psi[["item", "valor", "Status"]].rename(columns={"item": "Variável", "valor": "PSI"}),
        hide_index=True,
        use_container_width=True,
        column_config={"PSI": st.column_config.NumberColumn(format="%.3f")},
    )
    perf = last[last.tipo != "psi"].set_index("item").valor
    c2.metric(
        "ROC-AUC na safra mais recente",
        f"{perf.get('roc_auc_safra_recente', float('nan')):.3f}",
        delta=f"{perf.get('roc_auc_safra_recente', 0) - perf.get('roc_auc_referencia', 0):+.3f} vs. teste",
    )
    c2.metric(
        "Observado / esperado",
        f"{perf.get('observado_sobre_esperado', float('nan')):.2f}",
        help="Faixa aceitável: 0,67 a 1,5",
    )
    flag = bool(last.retrain.iloc[0])
    (c2.error if flag else c2.success)("⛔ Retreino recomendado" if flag else "✅ Modelo estável")
    c2.caption("PSI < 0,10 estável · 0,10–0,25 atenção · > 0,25 mudança relevante (Siddiqi, 2006).")

dec = load("decisions")
if not dec.empty:
    st.markdown("#### Trilha de auditoria: champion × challenger")
    st.dataframe(dec.sort_values("decided_at", ascending=False), hide_index=True, use_container_width=True)
