from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.diagnostic import breaks_cusumolsresid

import alphafade as af
from alphafade import _supwald_table
from alphafade.breaks import supwald_pvalue
from tests.conftest import month_ends


def shifted(n: int, k: int, shift: float, sd: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    y = rng.normal(0.01, sd, n)
    y[k:] += shift
    return pd.Series(y, index=month_ends(n))


# -- the simulated table --------------------------------------------------------------
def test_table_matches_published_andrews_critical_values() -> None:
    # Andrews (1993) Table 1, p=1, pi0=0.15: 7.17 / 8.85 / 12.35. Later corrected tables
    # (e.g. Stock & Watson) print 7.12 / 8.68 / 12.16. Ours must sit in that neighbourhood.
    levels = np.asarray(_supwald_table.LEVELS)
    q = np.asarray(_supwald_table.QUANTILES[0.15])
    cv = {lvl: float(np.interp(lvl, levels, q)) for lvl in (0.90, 0.95, 0.99)}
    assert 7.05 <= cv[0.90] <= 7.25
    assert 8.60 <= cv[0.95] <= 8.95
    assert 12.05 <= cv[0.99] <= 12.45


def test_table_is_monotone_and_trim_ordered() -> None:
    for q in _supwald_table.QUANTILES.values():
        assert np.all(np.diff(q) >= 0)
    # Less trimming = more candidate dates = bigger maxima.
    assert _supwald_table.QUANTILES[0.05][-5] > _supwald_table.QUANTILES[0.25][-5]


def test_supwald_pvalue_edges() -> None:
    assert supwald_pvalue(0.0, 0.15) == pytest.approx(0.99)
    assert supwald_pvalue(1e6, 0.15) == pytest.approx(0.0005)
    assert supwald_pvalue(8.76, 0.15) == pytest.approx(0.05, abs=0.005)


# -- find_break -------------------------------------------------------------------------
def test_sup_wald_locates_planted_break() -> None:
    # A shift of one standard deviation: large enough that the date is pinned down tightly.
    y = shifted(400, 250, -0.03, 0.03, seed=1)
    res = af.find_break(y)
    assert res.method == "sup_wald"
    assert res.p_value < 0.001
    assert abs(res.n_before - 250) <= 5
    assert res.change == pytest.approx(-0.03, abs=0.008)
    assert res.change_t < -3
    assert res.path is not None
    assert res.path.idxmax() == res.date
    assert res.n_obs == 400
    assert "significant" in res.summary()


@pytest.mark.parametrize("seed", range(5))
def test_sup_wald_location_is_stable_across_seeds(seed: int) -> None:
    y = shifted(300, 120, 0.03, 0.03, seed=seed)
    res = af.find_break(y)
    assert abs(res.n_before - 120) <= 8


def test_sup_wald_size_under_no_break() -> None:
    rejections = 0
    for seed in range(200):
        y = shifted(240, 0, 0.0, 0.03, seed=100 + seed)
        rejections += af.find_break(y).p_value < 0.05
    # Nominal 5% (10 of 200); allow for sampling error.
    assert 2 <= rejections <= 18


def test_no_break_summary_says_no_significant() -> None:
    res = af.find_break(shifted(240, 0, 0.0, 0.03, seed=5))
    assert res.p_value > 0.05
    assert "no significant" in res.summary()


def test_cusum_matches_statsmodels_without_hac() -> None:
    y = shifted(300, 150, -0.01, 0.03, seed=4)
    res = af.find_break(y, method="cusum", hac_lags=0)
    resid = y.to_numpy() - y.mean()
    stat, pval, _ = breaks_cusumolsresid(resid, ddof=0)
    assert res.stat == pytest.approx(stat, rel=1e-10)
    assert res.p_value == pytest.approx(pval, rel=1e-8)
    assert res.method == "cusum"
    assert abs(res.n_before - 150) <= 20
    assert res.path is not None
    assert "CUSUM" in res.summary()


def test_break_argument_validation() -> None:
    y = shifted(100, 50, 0.0, 0.03, seed=0)
    with pytest.raises(af.InputError, match="trim"):
        af.find_break(y, trim=0.3)
    with pytest.raises(af.InputError, match="method"):
        af.find_break(y, method="bai")  # type: ignore[arg-type]
    with pytest.raises(af.InsufficientDataError, match="at least 20"):
        af.find_break(y.iloc[:10])
    with pytest.raises(af.InputError, match="constant"):
        af.find_break(pd.Series(0.01, index=month_ends(50)))


def test_rolling_input_raises_lag_floor() -> None:
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.01, 0.03, 300), index=month_ends(300))
    rs = af.rolling_sharpe(r, window=24)
    res = af.find_break(rs)
    assert res.hac_lags >= 23


# -- chow_test ---------------------------------------------------------------------------
def test_chow_matches_statsmodels_dummy_regression() -> None:
    import statsmodels.api as sm

    y = shifted(200, 90, -0.02, 0.02, seed=5)
    res = af.chow_test(y, y.index[90], hac_lags=3)
    x = np.column_stack([np.ones(200), (np.arange(200) >= 90).astype(float)])
    ref = sm.OLS(y.to_numpy(), x).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    assert res.change == pytest.approx(ref.params[1], rel=1e-10)
    assert res.change_t == pytest.approx(ref.tvalues[1], rel=1e-10)
    assert res.stat == pytest.approx(ref.tvalues[1] ** 2, rel=1e-10)
    assert res.p_value < 0.01
    assert res.date == y.index[90]
    assert res.n_before == 90
    assert "Chow" in res.summary()


def test_chow_date_between_observations_rounds_forward() -> None:
    y = shifted(100, 50, 0.0, 0.03, seed=0)
    res = af.chow_test(y, "1984-02-15")
    assert res.date == pd.Timestamp("1984-02-29")


def test_chow_needs_both_sides() -> None:
    y = shifted(100, 50, 0.0, 0.03, seed=0)
    with pytest.raises(af.InsufficientDataError, match="each side"):
        af.chow_test(y, y.index[2])
    with pytest.raises(af.InputError, match="valid date"):
        af.chow_test(y, "someday")


def test_significant_threshold() -> None:
    res = af.find_break(shifted(240, 0, 0.0, 0.03, seed=3))
    assert res.significant(alpha=1.0)
