"""O gerador precisa ser reprodutível e respeitar as invariantes contábeis do contas a receber."""

import pandas as pd
import pytest

from credit_risk.generator import generate
from credit_risk.ingest import TITULOS_SCHEMA


@pytest.fixture(scope="module")
def small(tmp_path_factory, monkeypatch_module):
    out = tmp_path_factory.mktemp("gen") / "landing"
    cfg = generate.load_config(generate.settings.generator_config)
    cfg["scale"].update(n_clients=400, n_reps=20, n_products=200)
    monkeypatch_module.setattr(generate, "load_config", lambda *a, **k: cfg)
    generate.main(out_dir=out)
    return out


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_titulos_respect_contract(small):
    TITULOS_SCHEMA.validate(pd.read_parquet(small / "titulos.parquet"), lazy=True)


def test_accounting_invariants(small):
    t = pd.read_parquet(small / "titulos.parquet")
    notas = pd.read_parquet(small / "notas.parquet")
    assert t.titulo_id.is_unique
    assert ((t.situacao == "PAGO") == t.dt_pagamento.notna()).all()
    assert (t.loc[t.situacao == "ABERTO", "valor_pago"] == 0).all()
    # soma das parcelas == valor da nota (tolerância de arredondamento)
    soma = t.groupby("nota_id").valor_nominal.sum()
    assert (soma - notas.set_index("nota_id").valor_total.reindex(soma.index)).abs().max() < 0.05 * 6
    # TERC (cobrança terceirizada) só existe para títulos vencidos e não pagos (informação pós-evento)
    assert (t[t.tipo_atual == "TERC"].situacao == "ABERTO").all()


def test_ids_are_random_and_non_sequential(small):
    t = pd.read_parquet(small / "titulos.parquet")
    assert t.titulo_id.astype(str).str.len().eq(12).all()
    assert not t.titulo_id.sort_values().diff().dropna().eq(1).any()  # nada sequencial


def test_products_have_no_description(small):
    p = pd.read_parquet(small / "produtos.parquet")
    assert set(p.columns) == {"produto_id", "linha", "preco_unitario"}


def test_reproducible(small, tmp_path):
    generate.main(out_dir=tmp_path / "landing")
    a = pd.read_parquet(small / "titulos.parquet")
    b = pd.read_parquet(tmp_path / "landing" / "titulos.parquet")
    pd.testing.assert_frame_equal(a, b)
