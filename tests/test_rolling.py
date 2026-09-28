from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import alphafade as af
from alphafade.rolling import WINDOW_ATTR
from tests.conftest import business_days, month_ends

# --------------------------------------------------------------------------------------
# Hand-computed 3-asset, 4-date example (spec validation test)
#   d1: signal ranks (1,2,3), return ranks (1,3,2)  -> Spearman 1 - 6*2/(3*8) = 0.5
#   d2: identical ranking                           -> 1
#   d3: reversed ranking                            -> -1
#   d4: tied signal (2,2,1) -> ranks (2.5,2.5,1); returns ranks (1,2,3)
#       deviations (0.5,0.5,-1) and (-1,0,1): -1.5 / sqrt(1.5*2) = -sqrt(3)/2
# Pearson on d1: deviations (-1,0,1) and (-.01,.01,0) -> .01 / sqrt(2*.0002) = 0.5
# --------------------------------------------------------------------------------------
DATES4 = month_ends(4, "2020-01-31")
SIG4 = pd.DataFrame(
    [[1, 2, 3], [3, 1, 2], [1, 2, 3], [2, 2, 1]], index=DATES4, columns=["a", "b", "c"]
)
FWD4 = pd.DataFrame(
    [
        [0.01, 0.03, 0.02],
        [0.05, -0.02, 0.00],
        [0.03, 0.02, 0.01],
        [0.01, 0.02, 0.03],
    ],
    index=DATES4,
    columns=["a", "b", "c"],
)


def test_hand_computed_spearman_ic() -> None:
    ic = af.ic_series(SIG4, FWD4, min_assets=3)
    expected = [0.5, 1.0, -1.0, -np.sqrt(3) / 2]
    np.testing.assert_allclose(ic.to_numpy(), expected, rtol=0, atol=1e-12)
    assert ic.name == "ic"


def test_hand_computed_pearson_ic_first_date() -> None:
    ic = af.ic_series(SIG4, FWD4, method="pearson", min_assets=3)
    assert ic.iloc[0] == pytest.approx(0.5, abs=1e-12)


def test_min_assets_blocks_small_cross_sections() -> None:
    with pytest.warns(af.DataDroppedWarning, match="fewer than 5 usable assets"):
        ic = af.ic_series(SIG4, FWD4)  # default min_assets=5 > 3 assets
    assert ic.isna().all()


def test_constant_signal_date_is_nan_and_warned() -> None:
    sig = SIG4.copy()
    sig.iloc[1] = 7.0
    with pytest.warns(af.DataDroppedWarning, match="constant signal"):
        ic = af.ic_series(sig, FWD4, min_assets=3)
    assert np.isnan(ic.iloc[1]) and ic.iloc[0] == pytest.approx(0.5)


def test_missing_values_use_pairwise_complete_assets() -> None:
    sig = pd.DataFrame([[1.0, 2.0, 3.0, np.nan]], index=DATES4[:1], columns=["a", "b", "c", "d"])
    fwd = pd.DataFrame([[0.01, 0.03, 0.02, 0.5]], index=DATES4[:1], columns=["a", "b", "c", "d"])
    ic = af.ic_series(sig, fwd, min_assets=3)
    assert ic.iloc[0] == pytest.approx(0.5)


def test_rows_with_no_data_are_nan_without_warning() -> None:
    fwd = FWD4.copy()
    fwd.iloc[-1] = np.nan  # what forward_returns produces on the last date
    ic = af.ic_series(SIG4, fwd, min_assets=3)
    assert np.isnan(ic.iloc[-1])


def test_ic_argument_validation() -> None:
    with pytest.raises(af.InputError, match="spearman"):
        af.ic_series(SIG4, FWD4, method="kendall")  # type: ignore[arg-type]
    with pytest.raises(af.InputError, match="at least 2"):
        af.ic_series(SIG4, FWD4, min_assets=1)
    with pytest.raises(af.InputError, match="integer"):
        af.ic_series(SIG4, FWD4, min_assets=2.5)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Look-ahead: the most important alignment property.
# --------------------------------------------------------------------------------------
def test_forward_returns_prevent_look_ahead(rng: np.random.Generator) -> None:
    n_dates, n_assets = 240, 200
    realized = pd.DataFrame(rng.normal(0, 0.05, (n_dates, n_assets)), index=month_ends(n_dates))
    fwd = af.forward_returns(realized)
    # A signal equal to the *current* realized return is legitimately known at t and has
    # no predictive power for iid returns: IC must be ~0.
    ic_known = af.ic_series(realized, fwd)
    assert abs(ic_known.mean()) < 0.01
    # A signal equal to the forward return is look-ahead: IC must be exactly 1.
    ic_cheat = af.ic_series(fwd, fwd)
    np.testing.assert_allclose(ic_cheat.dropna().to_numpy(), 1.0)
    # And forward_returns really is "next period": row t == realized row t+1.
    np.testing.assert_allclose(fwd.iloc[:-1].to_numpy(), realized.iloc[1:].to_numpy())
    assert fwd.iloc[-1].isna().all()


def test_forward_returns_multi_period_compounds() -> None:
    r = pd.Series([0.10, 0.20, -0.50, 0.0], index=month_ends(4))
    fwd2 = af.forward_returns(r, periods=2)
    assert fwd2.iloc[0] == pytest.approx(1.2 * 0.5 - 1)
    assert fwd2.iloc[1] == pytest.approx(0.5 * 1.0 - 1)
    assert fwd2.iloc[2:].isna().all()


def test_forward_returns_total_loss_and_validation() -> None:
    r = pd.Series([0.0, -1.0, 0.1], index=month_ends(3))
    assert af.forward_returns(r).iloc[0] == pytest.approx(-1.0)
    with pytest.raises(af.InputError, match="below -1"):
        af.forward_returns(pd.Series([0.0, -5.0, 0.1], index=month_ends(3)))
    with pytest.raises(af.InputError, match="positive integer"):
        af.forward_returns(r, periods=0)


# --------------------------------------------------------------------------------------
# rolling_ic / rolling_sharpe
# --------------------------------------------------------------------------------------
def test_rolling_ic_is_rolling_mean_of_ic_series(rng: np.random.Generator) -> None:
    idx = month_ends(60)
    sig = pd.DataFrame(rng.normal(size=(60, 30)), index=idx)
    fwd = 0.2 * sig + pd.DataFrame(rng.normal(size=(60, 30)), index=idx)
    ic = af.ic_series(sig, fwd)
    ric = af.rolling_ic(sig, fwd, window=12)
    pd.testing.assert_series_equal(
        ric, ic.rolling(12).mean().rename("rolling_ic"), check_exact=False
    )
    assert ric.attrs[WINDOW_ATTR] == 12
    ric_mp = af.rolling_ic(sig, fwd, window=12, min_periods=6)
    assert ric_mp.notna().sum() == 55


def test_rolling_ic_rejects_mixed_frequency(rng: np.random.Generator) -> None:
    idx = pd.date_range("1990-01-31", periods=60, freq="ME").append(
        pd.bdate_range("1995-01-02", periods=2000)
    )
    sig = pd.DataFrame(rng.normal(size=(len(idx), 10)), index=idx)
    with pytest.raises(af.FrequencyError):
        af.rolling_ic(sig, sig, window=12)


def test_rolling_sharpe_matches_manual(rng: np.random.Generator) -> None:
    r = pd.Series(rng.normal(0.01, 0.05, 100), index=month_ends(100))
    sr = af.rolling_sharpe(r, window=24)
    last = r.iloc[-24:]
    expected = last.mean() / last.std(ddof=1) * np.sqrt(12)
    assert sr.iloc[-1] == pytest.approx(expected)
    assert sr.iloc[:23].isna().all() and sr.notna().sum() == 77
    assert sr.attrs[WINDOW_ATTR] == 24


def test_rolling_sharpe_daily_annualizes_with_252(rng: np.random.Generator) -> None:
    r = pd.Series(rng.normal(0.0005, 0.01, 400), index=business_days(400))
    sr = af.rolling_sharpe(r, window=252)
    last = r.iloc[-252:]
    assert sr.iloc[-1] == pytest.approx(last.mean() / last.std() * np.sqrt(252))
    sr_w = af.rolling_sharpe(r, window=252, freq="W")
    assert sr_w.iloc[-1] == pytest.approx(sr.iloc[-1] * np.sqrt(52 / 252))


def test_rolling_sharpe_risk_free(rng: np.random.Generator) -> None:
    idx = month_ends(50)
    r = pd.Series(rng.normal(0.01, 0.05, 50), index=idx)
    rf = pd.Series(0.002, index=idx)
    a = af.rolling_sharpe(r, window=12, rf=rf)
    b = af.rolling_sharpe(r, window=12, rf=0.002)
    pd.testing.assert_series_equal(a, b)
    with pytest.raises(af.InputError, match="rf is missing"):
        af.rolling_sharpe(r, window=12, rf=rf.iloc[:-5])


def test_rolling_sharpe_zero_vol_is_nan() -> None:
    r = pd.Series(0.01, index=month_ends(30))
    assert af.rolling_sharpe(r, window=12).isna().all()


def test_rolling_window_validation(rng: np.random.Generator) -> None:
    r = pd.Series(rng.normal(size=20), index=month_ends(20))
    with pytest.raises(af.InsufficientDataError):
        af.rolling_sharpe(r, window=21)
    with pytest.raises(af.InputError):
        af.rolling_sharpe(r, window=12, min_periods=13)


# --------------------------------------------------------------------------------------
# Property-based tests
# --------------------------------------------------------------------------------------
# Small integers as floats: plenty of ties (to exercise average ranking) and an exactly
# order-preserving transform below.
panel = arrays(
    np.float64,
    st.tuples(st.integers(3, 12), st.integers(5, 15)),
    elements=st.integers(-50, 50).map(float),
)


@settings(max_examples=60, deadline=None)
@given(panel, panel)
def test_ic_bounded_and_monotone_invariant(s_arr: np.ndarray, f_arr: np.ndarray) -> None:
    n = min(s_arr.shape[0], f_arr.shape[0])
    k = min(s_arr.shape[1], f_arr.shape[1])
    idx = month_ends(n)
    sig = pd.DataFrame(s_arr[:n, :k], index=idx)
    fwd = pd.DataFrame(f_arr[:n, :k], index=idx)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", af.DataDroppedWarning)
        ic = af.ic_series(sig, fwd, min_assets=3)
        # Spearman IC only depends on ranks: a strictly increasing transform can't change it.
        ic_t = af.ic_series(np.exp(sig / 10), fwd, min_assets=3)
        ic_neg = af.ic_series(-sig, fwd, min_assets=3)
    valid = ic.dropna()
    assert ((valid >= -1) & (valid <= 1)).all()
    np.testing.assert_allclose(ic.to_numpy(), ic_t.to_numpy(), atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(ic.to_numpy(), -ic_neg.to_numpy(), atol=1e-9, equal_nan=True)


@settings(max_examples=40, deadline=None)
@given(
    arrays(
        np.float64,
        st.integers(30, 80),
        # Realistic magnitudes: pandas' streaming rolling sums keep ~1e-17 residue, which
        # only matters for returns on the order of machine epsilon.
        elements=st.floats(-0.2, 0.2, allow_nan=False).filter(lambda x: x == 0 or abs(x) > 1e-8),
    ),
    st.floats(0.1, 10),
)
def test_sharpe_scale_invariant(r_arr: np.ndarray, scale: float) -> None:
    r = pd.Series(r_arr, index=month_ends(len(r_arr)))
    a = af.rolling_sharpe(r, window=12)
    b = af.rolling_sharpe(r * scale, window=12)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-6, atol=1e-8, equal_nan=True)
