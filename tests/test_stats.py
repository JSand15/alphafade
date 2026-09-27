from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

import alphafade as af
from alphafade import _stats


@pytest.mark.parametrize("lags", [0, 1, 4, 12])
def test_ols_hac_matches_statsmodels(rng: np.random.Generator, lags: int) -> None:
    n = 300
    e = np.zeros(n)
    for t in range(1, n):  # AR(1) errors so the HAC correction matters
        e[t] = 0.6 * e[t - 1] + rng.normal()
    x = np.column_stack([np.ones(n), rng.normal(size=n), (np.arange(n) > 150).astype(float)])
    y = x @ np.array([0.5, -1.0, 0.3]) + e
    ours = _stats.ols_hac(y, x, lags)
    ref = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    np.testing.assert_allclose(ours.params, ref.params, rtol=1e-10)
    np.testing.assert_allclose(ours.bse, ref.bse, rtol=1e-10)
    np.testing.assert_allclose(ours.tvalues, ref.tvalues, rtol=1e-10)


def test_hac_mean(rng: np.random.Generator) -> None:
    y = rng.normal(0.1, 1.0, 200)
    mean, se, t = _stats.hac_mean(y, 5)
    ref = sm.OLS(y, np.ones(200)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    assert mean == pytest.approx(y.mean())
    assert se == pytest.approx(ref.bse[0], rel=1e-10)
    assert t == pytest.approx(mean / se)


def test_ols_hac_rejects_singular_and_tiny() -> None:
    x = np.column_stack([np.ones(10), np.ones(10)])
    with pytest.raises(af.InsufficientDataError, match="singular"):
        _stats.ols_hac(np.arange(10.0), x, 0)
    with pytest.raises(af.InsufficientDataError, match="more observations"):
        _stats.ols_hac(np.arange(2.0), np.ones((2, 2)), 0)


def test_lag_rules() -> None:
    assert _stats.default_hac_lags(100) == 4
    assert _stats.default_hac_lags(1000) == 6
    assert _stats.resolve_hac_lags(None, 100, min_lags=35) == 35
    assert _stats.resolve_hac_lags(3, 100) == 3
    for bad in (-1, 2.5, True):
        with pytest.raises(af.InputError):
            _stats.resolve_hac_lags(bad, 100)  # type: ignore[arg-type]
    with pytest.raises(af.InputError, match="smaller than the sample"):
        _stats.resolve_hac_lags(100, 100)


def test_resolve_rng_reproducible() -> None:
    a = _stats.resolve_rng(7).normal(size=3)
    b = _stats.resolve_rng(7).normal(size=3)
    np.testing.assert_array_equal(a, b)
    g = np.random.default_rng(1)
    assert _stats.resolve_rng(g) is g
    with pytest.raises(af.InputError, match="rng"):
        _stats.resolve_rng("seed")  # type: ignore[arg-type]


def test_block_bootstrap_indices(rng: np.random.Generator) -> None:
    idx = _stats.block_bootstrap_indices(103, 10, rng)
    assert len(idx) == 103 and idx.min() >= 0 and idx.max() <= 102
    # Within each block, positions are consecutive.
    blocks = idx[:100].reshape(10, 10)
    assert (np.diff(blocks, axis=1) == 1).all()
    assert _stats.default_block_size(1000) == 17
    assert _stats.default_block_size(1000, min_block=36) == 36
    assert _stats.default_block_size(5, min_block=36) == 5
