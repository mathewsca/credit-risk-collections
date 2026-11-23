"""Gerador de dados sintéticos de contas a receber com mecanismo causal conhecido.

Simula o sistema transacional de uma distribuidora atacadista: filiais, representantes, clientes,
produtos, notas, itens e títulos a receber. O atraso > 30 dias segue um mecanismo explícito
(config/generator.yaml), o que permite medir o quão perto o modelo de ML chega do classificador
"oráculo" (PD verdadeira).

Privacidade por construção: IDs aleatórios (não sequenciais), nomes fictícios, produtos apenas
por código, geografia e calendário próprios e perturbação de ±1–5% nos valores.

Uso:
    python -m credit_risk.generator.generate                  # cenário base
    python -m credit_risk.generator.generate --scenario shock # choque macro a partir de out/2024
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from faker import Faker

from credit_risk.settings import settings

CITIES = {
    "GO": ["Goiânia", "Anápolis", "Rio Verde", "Aparecida de Goiânia", "Catalão", "Itumbiara"],
    "MG": ["Uberlândia", "Uberaba", "Belo Horizonte", "Montes Claros", "Patos de Minas"],
    "DF": ["Brasília", "Taguatinga"],
    "MT": ["Cuiabá", "Rondonópolis", "Sinop"],
    "MS": ["Campo Grande", "Dourados"],
    "SP": ["Ribeirão Preto", "São José do Rio Preto", "Franca"],
    "TO": ["Palmas", "Araguaína"],
    "BA": ["Barreiras", "Luís Eduardo Magalhães"],
    "RO": ["Porto Velho", "Ji-Paraná"],
    "PA": ["Marabá", "Redenção"],
    "AM": ["Manaus"],
    "AC": ["Rio Branco"],
}
BRANCH_UF_P = {
    "GO": 0.43, "MG": 0.16, "DF": 0.07, "MT": 0.07, "MS": 0.07,
    "SP": 0.05, "TO": 0.05, "BA": 0.04, "RO": 0.03, "PA": 0.03,
}  # fmt: skip
SEGMENTOS = ["Comércio", "Distribuição", "Atacado", "Mercantil", "Center", "Varejo", "Suprimentos", "Store"]
SUFIXOS = ["Ltda", "ME", "EIRELI", "Comércio Ltda", "& Cia", "S.A."]

# Quantidade de dígitos de cada chave sintética. Propositalmente diferente de formatos de ERPs comuns:
# os IDs são aleatórios (não sequenciais) e não podem coincidir com códigos reais numa busca no sistema.
ID_DIGITS = {
    "filial": 4, "representante": 7, "cliente": 10, "produto": 8,
    "nota": 11, "numero_documento": 9, "titulo": 12,
}  # fmt: skip


def _choice(rng: np.random.Generator, p: dict, size: int) -> np.ndarray:
    keys = list(p)
    probs = np.array([p[k] for k in keys], dtype=float)
    return rng.choice(np.array(keys, dtype=object), size=size, p=probs / probs.sum())


def _ids(rng: np.random.Generator, n: int, kind: str) -> np.ndarray:
    """n inteiros únicos e aleatórios com ID_DIGITS[kind] dígitos."""
    d = ID_DIGITS[kind]
    draw = pd.unique(rng.integers(10 ** (d - 1), 10**d, size=int(n * 1.2) + 16))
    assert len(draw) >= n, "colisão excessiva de IDs"
    return draw[:n].astype(np.int64)


def _client_names(rng: np.random.Generator, fake: Faker, n: int) -> list[str]:
    """Razão social fictícia composta (sobrenome + segmento + sufixo): evita coincidência com nomes reais."""
    return [f"{fake.last_name()} {rng.choice(SEGMENTOS)} {rng.choice(SUFIXOS)}" for _ in range(n)]


def load_config(path: Path, scenario: str = "base") -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if scenario == "shock":
        # Choque: inflação de ticket (+25%) e deterioração de crédito em GO/MG a partir de out/2024
        cfg["risk"]["shock"] = {
            "start": "2024-10-01",
            "intercept": 1.0,
            "uf": {"GO": 0.6, "MG": 0.5},
            "value_mult": 1.25,
        }
    return cfg


def gen_dimensions(cfg: dict, rng: np.random.Generator, fake: Faker) -> dict[str, pd.DataFrame]:
    sc = cfg["scale"]
    start = pd.Timestamp(cfg["period"]["start"])

    # Filiais
    n_b = sc["n_branches"]
    b_uf = _choice(rng, BRANCH_UF_P, n_b)
    b_city = [rng.choice(CITIES[u]) for u in b_uf]
    filiais = pd.DataFrame(
        {
            "filial_id": _ids(rng, n_b, "filial"),
            "nome_filial": [f"Filial {c} {chr(65 + i % 26)}{i // 26 + 1}" for i, c in enumerate(b_city)],
            "divisao": np.where(np.arange(n_b) < int(n_b * 0.8), "Atacado", "Varejo"),
            "uf": b_uf,
            "cidade": b_city,
        }
    )

    # Representantes (vinculados a uma filial)
    n_r = sc["n_reps"]
    representantes = pd.DataFrame(
        {
            "representante_id": _ids(rng, n_r, "representante"),
            "nome": [fake.name() for _ in range(n_r)],
            "filial_id": rng.choice(filiais.filial_id, size=n_r),
            "dt_admissao": [start - pd.Timedelta(days=int(d)) for d in rng.integers(30, 3650, n_r)],
        }
    )

    # Clientes
    n_c = sc["n_clients"]
    c_uf = _choice(rng, cfg["clients"]["uf_p"], n_c)
    since = start + pd.to_timedelta(rng.integers(-14 * 365, 600, n_c), unit="D")
    clientes = pd.DataFrame(
        {
            "cliente_id": _ids(rng, n_c, "cliente"),
            "razao_social": _client_names(rng, fake, n_c),
            "porte": _choice(rng, cfg["clients"]["size_p"], n_c),
            "uf": c_uf,
            "cidade": [rng.choice(CITIES.get(u, ["Capital"])) for u in c_uf],
            "cliente_desde": since.normalize(),
        }
    )
    # carteira: cliente atendido por filial do mesmo UF quando existir
    by_uf = filiais.groupby("uf").filial_id.apply(list).to_dict()
    clientes["filial_id"] = [rng.choice(by_uf.get(u, filiais.filial_id.tolist())) for u in c_uf]
    reps_by_b = representantes.groupby("filial_id").representante_id.apply(list).to_dict()
    all_reps = representantes.representante_id.tolist()
    clientes["representante_id"] = [rng.choice(reps_by_b.get(b, all_reps)) for b in clientes.filial_id]

    # Produtos: somente código + linha genérica (sem descrição, para não revelar o segmento)
    n_p = sc["n_products"]
    produtos = pd.DataFrame(
        {
            "produto_id": _ids(rng, n_p, "produto"),
            "linha": _choice(rng, cfg["products"]["lines"], n_p),
            "preco_unitario": np.round(rng.lognormal(np.log(45), 0.9, n_p), 2),
        }
    )
    return {"filiais": filiais, "representantes": representantes, "clientes": clientes, "produtos": produtos}


def gen_invoices(cfg: dict, rng: np.random.Generator, dims: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    sc, inv = cfg["scale"], cfg["invoice"]
    cl = dims["clientes"]
    months = pd.date_range(cfg["period"]["start"], cfg["period"]["end"], freq="MS")
    rate = rng.gamma(
        sc["monthly_orders_shape"], sc["monthly_orders_mean"] / sc["monthly_orders_shape"], len(cl)
    )
    shock = cfg["risk"]["shock"]
    rows = []
    for m_idx, m in enumerate(months):
        lam = rate * (1 + sc["growth_per_month"]) ** m_idx
        n = rng.poisson(lam)
        n[(cl.cliente_desde > m + pd.offsets.MonthEnd(0)).to_numpy()] = 0
        idx = np.repeat(np.arange(len(cl)), n)
        if idx.size == 0:
            continue
        days = rng.integers(0, m.days_in_month, idx.size)
        rows.append(pd.DataFrame({"cli_idx": idx, "dt_emissao": m + pd.to_timedelta(days, unit="D")}))
    notas = pd.concat(rows, ignore_index=True).sort_values("dt_emissao", ignore_index=True)
    notas["nota_id"] = _ids(rng, len(notas), "nota")
    notas["numero_documento"] = _ids(rng, len(notas), "numero_documento")
    c = cl.iloc[notas.cli_idx.to_numpy()].reset_index(drop=True)
    notas["cliente_id"] = c.cliente_id.to_numpy()
    notas["filial_id"] = c.filial_id.to_numpy()
    notas["representante_id"] = c.representante_id.to_numpy().astype(float)
    notas.loc[rng.random(len(notas)) < 0.08, "representante_id"] = np.nan  # venda de balcão
    mult = np.where(notas.dt_emissao >= pd.Timestamp(shock["start"]), shock.get("value_mult", 1.0), 1.0)
    valor = rng.lognormal(np.log(inv["value_median"]), inv["value_sigma"], len(notas)) * mult
    # Perturbação anti-rastreio: fator 1 ± U(lo, hi) com sinal aleatório por nota
    lo, hi = inv.get("privacy_jitter", [0.0, 0.0])
    jitter = 1 + rng.choice([-1, 1], len(notas)) * rng.uniform(lo, hi, len(notas))
    notas["valor_total"] = np.round(valor * jitter, 2)
    notas["qtd_parcelas"] = _choice(rng, inv["installments_p"], len(notas)).astype(int)
    notas["prazo_inicial"] = rng.choice(inv["first_due_days"], len(notas))
    notas["tipo"] = _choice(rng, inv["types_p"], len(notas))

    # Itens da nota (produto com popularidade Zipf)
    n_items = np.clip(
        np.round(rng.lognormal(np.log(inv["items_median"]), inv["items_sigma"], len(notas))), 1, 60
    ).astype(int)
    n_prod = len(dims["produtos"])
    pz = np.arange(1, n_prod + 1) ** (-cfg["products"]["zipf_a"])
    perm = dims["produtos"].produto_id.to_numpy()[rng.permutation(n_prod)]
    prod = perm[rng.choice(n_prod, size=n_items.sum(), p=pz / pz.sum())]
    itens = pd.DataFrame({"nota_id": np.repeat(notas.nota_id.to_numpy(), n_items), "produto_id": prod})
    itens = itens.drop_duplicates(ignore_index=True)
    itens["quantidade"] = rng.integers(1, 12, len(itens))
    return notas.drop(columns="cli_idx"), itens


def gen_titles(cfg: dict, rng: np.random.Generator, notas: pd.DataFrame, dims: dict) -> pd.DataFrame:
    rk, pay, inv = cfg["risk"], cfg["payment"], cfg["invoice"]
    snapshot = pd.Timestamp(cfg["period"]["snapshot_date"])
    t = notas.loc[notas.index.repeat(notas.qtd_parcelas.to_numpy())].reset_index(drop=True)
    t["parcela"] = t.groupby("nota_id").cumcount() + 1
    t["dt_vencimento"] = t.dt_emissao + pd.to_timedelta(t.prazo_inicial + 30 * (t.parcela - 1), unit="D")
    t["valor_nominal"] = np.round(t.valor_total / t.qtd_parcelas, 2)
    cl = dims["clientes"]
    t = t.merge(cl[["cliente_id", "porte", "uf", "cliente_desde"]], on="cliente_id", how="left")
    t = t.sort_values(["cliente_id", "dt_vencimento"], ignore_index=True)
    n_cli = len(cl)
    cli_pos = pd.Index(cl.cliente_id).get_indexer(t.cliente_id)
    shock = rk["shock"]

    # --- 1) regime mensal do cliente: cadeia de Markov adimplente <-> em dificuldade ---
    months = pd.date_range(cfg["period"]["start"], t.dt_vencimento.max(), freq="MS")
    distressed = rng.random(n_cli) < cfg["clients"]["distressed_share"]
    static = (
        rk["hazard_intercept"]
        + rk["distressed"] * distressed
        + rng.normal(0, rk["latent_sd"], n_cli)
        + cl.porte.map(rk["size"]).fillna(0).to_numpy()
        + cl.uf.map(rk["uf_high_risk"]).fillna(0).to_numpy()
    )
    shock_uf = cl.uf.map(shock.get("uf", {})).fillna(0).to_numpy()
    regime = np.zeros((n_cli, len(months)), dtype=bool)
    state = np.zeros(n_cli, dtype=bool)
    for j, m in enumerate(months):
        tenure = np.clip((m - cl.cliente_desde).dt.days.to_numpy() / 365.25, 0, 15)
        logit_h = static + rk["tenure_years"] * tenure + rk["q1_seasonality"] * (m.month <= 3)
        if m >= pd.Timestamp(shock["start"]):
            logit_h = logit_h + shock.get("intercept", 0) + shock_uf
        enter = rng.random(n_cli) < 1 / (1 + np.exp(-logit_h))
        stay = rng.random(n_cli) < rk["persist"]
        state = np.where(state, stay, enter)
        regime[:, j] = state
    m_idx = (
        (t.dt_vencimento.dt.year - months[0].year) * 12 + t.dt_vencimento.dt.month - months[0].month
    ).to_numpy()
    in_distress = regime[cli_pos, m_idx]

    # --- 2) atraso pontual no regime adimplente (características do título) ---
    nota_codes, nota_uniques = pd.factorize(t.nota_id)
    nota_effect = rng.normal(0, rk["invoice_sd"], len(nota_uniques))[nota_codes]
    title_logit = (
        rk["title_intercept"]
        + nota_effect
        + rk["log_value"] * np.log(t.valor_nominal / (inv["value_median"] / 2.2))
        + rk["installment_number"] * (t.parcela - 1)
        + t.tipo.map(rk["types"]).fillna(0)
    ).to_numpy()
    pd_true = np.where(in_distress, rk["p_late_distress"], 1 / (1 + np.exp(-title_logit)))
    late = rng.random(len(t)) < pd_true
    t["pd_true"], t["late30_true"] = pd_true, late

    # --- 3) dias de pagamento ---
    n = len(t)
    delay = np.empty(n)
    ok = ~late
    on_time = ok & (rng.random(n) < pay["on_time_share"])
    delay[on_time] = -np.minimum(rng.geometric(1 / pay["early_days_mean"], on_time.sum()) - 1, 40)
    short = ok & ~on_time
    delay[short] = np.minimum(rng.geometric(0.22, short.sum()), pay["short_late_max"])
    mid = late & (rng.random(n) < pay["late_31_90_share"])
    delay[mid] = rng.integers(31, 91, mid.sum())
    long_ = late & ~mid
    delay[long_] = rng.integers(91, 400, long_.sum())
    delay[long_ & (rng.random(n) < pay["never_paid_share"])] = np.nan

    t["dt_pagamento"] = t.dt_vencimento + pd.to_timedelta(delay, unit="D")
    t.loc[t.dt_pagamento > snapshot, "dt_pagamento"] = pd.NaT  # ainda não pago na data do snapshot
    paid = t.dt_pagamento.notna()
    t["situacao"] = np.where(paid, "PAGO", "ABERTO")
    t["valor_pago"] = np.where(paid, t.valor_nominal, 0.0)
    d_paid = (t.dt_pagamento - t.dt_vencimento).dt.days
    charged = rng.random(n) < inv.get("late_fee_charge_prob", 1.0)
    t["valor_juros"] = np.where(
        paid & (d_paid > 0) & charged, np.round(t.valor_nominal * inv["late_fee_daily"] * d_paid, 2), 0.0
    )
    # tipo atual: vencidos há > 90d sem pagamento vão para cobrança terceirizada (informação PÓS-evento!)
    t["tipo_origem"] = t.tipo
    overdue = (~paid) & ((snapshot - t.dt_vencimento).dt.days > pay["outsourced_after_days"])
    t["tipo_atual"] = np.where(overdue, "TERC", t.tipo)
    t["titulo_id"] = _ids(rng, n, "titulo")
    return t


def main(scenario: str = "base", out_dir: Path | None = None) -> dict[str, int]:
    cfg = load_config(settings.generator_config, scenario)
    rng = np.random.default_rng(cfg["seed"])
    fake = Faker("pt_BR")
    Faker.seed(cfg["seed"])
    out = Path(out_dir) if out_dir else settings.landing_dir
    truth = out.parent / "ground_truth"
    out.mkdir(parents=True, exist_ok=True)
    truth.mkdir(parents=True, exist_ok=True)

    dims = gen_dimensions(cfg, rng, fake)
    notas, itens = gen_invoices(cfg, rng, dims)
    titulos = gen_titles(cfg, rng, notas, dims)

    tables = {
        **dims,
        "notas": notas[["nota_id", "numero_documento", "filial_id", "cliente_id", "representante_id",
                        "dt_emissao", "valor_total", "qtd_parcelas"]],
        "notas_itens": itens,
        "titulos": titulos[["titulo_id", "nota_id", "filial_id", "cliente_id", "numero_documento", "parcela",
                            "tipo_origem", "tipo_atual", "dt_emissao", "dt_vencimento", "dt_pagamento",
                            "valor_nominal", "valor_juros", "valor_pago", "situacao"]],
    }  # fmt: skip
    for name, df in tables.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
    titulos[["titulo_id", "pd_true", "late30_true"]].to_parquet(
        truth / "titulos_pd_true.parquet", index=False
    )
    (out / "_scenario.txt").write_text(scenario, encoding="utf-8")
    counts = {k: len(v) for k, v in tables.items()}
    print(f"[generator] cenário={scenario} -> {out}")
    for k, v in counts.items():
        print(f"  {k:<15} {v:>9,}")
    print(f"  taxa late30 (verdade): {titulos.late30_true.mean():.2%}")
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["base", "shock"], default="base")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    main(a.scenario, a.out)
