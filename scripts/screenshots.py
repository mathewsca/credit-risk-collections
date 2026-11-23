"""Captura screenshots das páginas do app Streamlit para o README.

Uso (com o app rodando em localhost:8501):
    pip install playwright && playwright install chromium
    python scripts/screenshots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
PAGES = {
    "01_visao_geral": "",
    "02_carteira": "Carteira_e_Representantes",
    "03_titulos": "Titulos",
    "04_risco_cobranca": "Risco_e_Cobranca",
    "05_monitoramento": "Monitoramento_do_Modelo",
}
CONTENT = "[data-testid='stMetric'], [data-testid='stDataFrame'], .js-plotly-plot"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        for name, path in PAGES.items():
            page.goto(f"{BASE}/{path}", wait_until="networkidle")
            page.wait_for_selector(CONTENT, timeout=120_000)  # espera o conteúdo, não só o "esqueleto"
            page.wait_for_timeout(5000)  # gráficos Plotly terminam de renderizar
            page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
            print("ok", name)
        browser.close()


if __name__ == "__main__":
    main()
