import numpy as np
import pytest

from credit_risk.ml.metrics import bootstrap_ci, evaluate, ks_statistic, psi, psi_categorical, psi_status


def test_ks_perfect_and_random():
    y = np.array([0] * 50 + [1] * 50)
    assert ks_statistic(y, np.linspace(0, 1, 100)) == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    assert ks_statistic(rng.integers(0, 2, 5000), rng.random(5000)) < 0.06


def test_psi_identical_is_zero_and_shift_is_large():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 20000)
    assert psi(a, rng.normal(0, 1, 20000)) < 0.01
    assert psi(a, rng.normal(1.0, 1, 20000)) > 0.25


def test_psi_categorical_and_status():
    assert psi_categorical(["a"] * 50 + ["b"] * 50, ["a"] * 50 + ["b"] * 50) == pytest.approx(0.0)
    assert psi_categorical(["a"] * 90 + ["b"] * 10, ["a"] * 10 + ["b"] * 90) > 0.25
    assert psi_status(0.05) == "estável"
    assert psi_status(0.2) == "atenção"
    assert psi_status(0.4) == "mudança relevante"


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.3 + rng.random(2000) * 0.7, 0, 1)
    point = evaluate(y, p)
    ci = bootstrap_ci(y, p, n_boot=200)
    for k in ("roc_auc", "pr_auc", "brier", "ks"):
        assert ci[k][0] <= point[k] <= ci[k][1]


def test_paired_bootstrap_detects_real_difference_only():
    from credit_risk.ml.metrics import paired_bootstrap_diff

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 3000)
    good = np.clip(y * 0.4 + rng.random(3000) * 0.6, 0, 1)
    noise = rng.random(3000)
    better = paired_bootstrap_diff(y, good, noise, n_boot=300)
    assert better["ci_low"] > 0 and better["p_a_melhor"] > 0.99
    same = paired_bootstrap_diff(y, good, good + 1e-9 * rng.random(3000), n_boot=300)
    assert same["ci_low"] <= 0 <= same["ci_high"]
