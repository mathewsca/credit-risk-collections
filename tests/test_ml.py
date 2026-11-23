import numpy as np

from credit_risk.ml.features import FEATURES, load_features, temporal_split
from credit_risk.ml.promote import decide
from credit_risk.ml.score import risk_band, score_portfolio
from credit_risk.ml.train import build_baseline, build_lgbm


def test_temporal_split_is_ordered_and_disjoint(features_df):
    s = temporal_split(load_features(features_df))
    assert s.train.safra_vencimento.max() < s.valid.safra_vencimento.min()
    assert s.valid.safra_vencimento.max() < s.test.safra_vencimento.min()
    assert s.test.label_disponivel.all()
    ids = [set(x.cod_titulo) for x in (s.train, s.valid, s.test)]
    assert not (ids[0] & ids[1] or ids[1] & ids[2] or ids[0] & ids[2])


def test_models_learn_known_signal(features_df):
    from sklearn.metrics import roc_auc_score

    s = temporal_split(load_features(features_df))
    y = s.train.is_late30.astype(int)
    for model in (build_baseline(), build_lgbm(n_estimators=100)):
        model.fit(s.train[FEATURES], y)
        auc = roc_auc_score(s.test.is_late30.astype(int), model.predict_proba(s.test[FEATURES])[:, 1])
        assert auc > 0.70, type(model[-1]).__name__


def test_risk_bands_cover_all_values():
    bands = risk_band(np.array([0.0, 0.009, 0.02, 0.05, 0.2, 0.5, 1.0]))
    assert list(bands) == ["A", "A", "B", "C", "D", "E", "E"]


def test_score_portfolio_orders_by_expected_loss(features_df):
    class Const:
        def predict_proba(self, X):
            p = np.linspace(0.01, 0.5, len(X))
            return np.c_[1 - p, p]

    out = score_portfolio(load_features(features_df), Const())
    assert (out.status == "A").all() and (~out.is_late30).all()
    assert out.perda_esperada.is_monotonic_decreasing
    assert out.prioridade.iloc[0] == 1


def test_promotion_gate():
    good = {"roc_auc": 0.85, "pr_auc": 0.40, "brier": 0.015}
    assert decide(good, None)[0]
    assert not decide({**good, "roc_auc": 0.6}, None)[0]
    assert not decide({**good, "pr_auc": 0.30}, good)[0]
    assert not decide({**good, "brier": 0.02}, good)[0]
    assert decide({**good, "pr_auc": 0.398}, good)[0]  # dentro da tolerância


def test_redundant_features_are_out_of_the_model():
    from credit_risk.ml.features import REDUNDANT

    assert not set(REDUNDANT) & set(FEATURES)
    assert set(REDUNDANT.values()) <= set(FEATURES)


def test_predict_pd_uses_model_own_feature_contract(features_df):
    from credit_risk.ml.registry import predict_pd

    df = load_features(features_df)
    cols = FEATURES[:5]
    model = build_baseline()
    model.set_params(pre__transformers=[("num", "passthrough", cols)])
    model.fit(df[cols], df.is_late30.astype(int))
    p = predict_pd(model, df)  # recebe o DataFrame completo, usa só as colunas do modelo
    assert len(p) == len(df) and ((p >= 0) & (p <= 1)).all()
