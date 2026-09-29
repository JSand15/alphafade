from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from alphafade import (
    DataDroppedWarning,
    InputError,
    InsufficientDataError,
    forward_returns,
    ic_series,
)
from alphafade._stats import default_hac_lags, hac_mean
from alphafade.horizon import HorizonResult, _fit_half_life, ic_by_horizon
from tests.conftest import month_ends

RHO = 0.3


def one_period_panel(
    n_dates: int = 240, n_assets: int = 60, rho: float = RHO, seed: int = 7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Signal known at t drives ONLY the return earned in (t, t+1]; later returns are noise."""
    g = np.random.default_rng(seed)
    dates = month_ends(n_dates)
    cols = [f"a{i}" for i in range(n_assets)]
    sig = g.standard_normal((n_dates, n_assets))
    ret = 0.05 * (
        rho * np.roll(sig, 1, axis=0) + math.sqrt(1 - rho**2) * g.standard_normal(sig.shape)
    )
    ret[0] = 0.05 * g.standard_normal(n_assets)
    return pd.DataFrame(sig, dates, cols), pd.DataFrame(ret, dates, cols)


def persistent_panel(
    n_dates: int = 240, n_assets: int = 60, phi: float = 0.97, seed: int = 11
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A slow-moving signal: every future period's return keeps loading on it."""
    g = np.random.default_rng(seed)
    dates = month_ends(n_dates)
    cols = [f"a{i}" for i in range(n_assets)]
    sig = np.zeros((n_dates, n_assets))
    sig[0] = g.standard_normal(n_assets)
    for t in range(1, n_dates):
        sig[t] = phi * sig[t - 1] + math.sqrt(1 - phi**2) * g.standard_normal(n_assets)
    ret = np.zeros_like(sig)
    ret[1:] = 0.05 * (0.2 * sig[:-1] + g.standard_normal((n_dates - 1, n_assets)))
    ret[0] = 0.05 * g.standard_normal(n_assets)
    return pd.DataFrame(sig, dates, cols), pd.DataFrame(ret, dates, cols)


def test_h1_near_rho_and_falls_like_root_h() -> None:
    sig, ret = one_period_panel()
    res = ic_by_horizon(sig, ret)
    t = res.table
    assert list(t.index) == [1, 2, 3, 6, 12]
    assert t.loc[1, "mean_ic"] == pytest.approx(RHO, abs=0.03)
    # Cumulative windows include one informative period out of h, so IC ~ rho / sqrt(h).
    for h in (2, 3, 6, 12):
        assert t.loc[h, "mean_ic"] == pytest.approx(RHO / math.sqrt(h), abs=0.03)
    # The information added after h=1 is nil: IC * sqrt(h) stays flat.
    scaled = t["mean_ic"] * np.sqrt(t.index.to_numpy(dtype=float))
    assert scaled.max() - scaled.min() < 0.08
    assert res.peak_horizon == 1
    assert t.loc[1, "t_stat"] > 10
    assert res.half_life_periods is not None
    assert 2.0 < res.half_life_periods < 20.0
    assert res.reason is None


def test_persistent_signal_ic_grows_with_horizon() -> None:
    sig, ret = persistent_panel()
    res = ic_by_horizon(sig, ret, horizons=(1, 3, 6, 12))
    ic = res.table["mean_ic"]
    assert ic.loc[6] > ic.loc[1] and ic.loc[12] > ic.loc[3] > ic.loc[1]
    assert res.peak_horizon in (6, 12)
    assert res.half_life_periods is None
    assert res.reason is not None and "not decreasing" in res.reason
    assert "no half-life" in res.summary().lower()


def test_h1_matches_ic_series_exactly() -> None:
    sig, ret = one_period_panel(n_dates=80, n_assets=20)
    res = ic_by_horizon(sig, ret, horizons=[1, 2, 4])
    direct = ic_series(sig, forward_returns(ret, 1)).mean()
    assert res.table.loc[1, "mean_ic"] == direct
    assert res.table.loc[1, "n_dates"] == 79
    for h in (2, 4):
        raw = ic_series(sig, forward_returns(ret, h))
        ic = raw.dropna()
        assert res.table.loc[h, "mean_ic"] == raw.mean()
        assert res.table.loc[h, "n_dates"] == len(ic)
        lags = max(default_hac_lags(len(ic)), h - 1)
        assert res.table.loc[h, "t_stat"] == pytest.approx(hac_mean(ic.to_numpy(), lags)[2])
        assert res.table.loc[h, "ic_per_period"] == pytest.approx(ic.mean() / h)


def test_pearson_method_and_hac_lags_override() -> None:
    sig, ret = one_period_panel(n_dates=80, n_assets=20)
    res = ic_by_horizon(sig, ret, horizons=[1, 3], method="pearson", hac_lags=2)
    ic = ic_series(sig, forward_returns(ret, 3), method="pearson").dropna()
    assert res.table.loc[3, "t_stat"] == pytest.approx(hac_mean(ic.to_numpy(), 2)[2])
    ic_sp = ic_by_horizon(sig, ret, horizons=[1, 3]).table.loc[3, "mean_ic"]
    assert res.table.loc[3, "mean_ic"] != ic_sp


def test_horizons_sorted_and_accept_numpy_ints() -> None:
    sig, ret = one_period_panel(n_dates=60, n_assets=15)
    res = ic_by_horizon(sig, ret, horizons=np.array([6, 1, 3]))
    assert list(res.table.index) == [1, 3, 6]
    assert res.table.index.name == "horizon"
    assert list(res.table.columns) == ["mean_ic", "t_stat", "n_dates", "ic_per_period"]


def test_no_lookahead_contemporaneous_signal_has_no_skill() -> None:
    g = np.random.default_rng(3)
    dates = month_ends(200)
    ret = pd.DataFrame(0.05 * g.standard_normal((200, 40)), dates)
    # Signal = the return already realized at t: known at t, says nothing about (t, t+h].
    res = ic_by_horizon(ret, ret, horizons=(1, 2, 3))
    assert (res.table["mean_ic"].abs() < 0.05).all()
    assert (res.table["t_stat"].abs() < 3.5).all()


def test_no_lookahead_deliberate_peek_is_detected() -> None:
    g = np.random.default_rng(3)
    dates = month_ends(200)
    ret = pd.DataFrame(0.05 * g.standard_normal((200, 40)), dates)
    peek = ret.shift(-1)  # tomorrow's return used as today's signal
    res = ic_by_horizon(peek, ret, horizons=(1,))
    assert res.table.loc[1, "mean_ic"] > 0.99


def test_future_data_after_window_cannot_change_earlier_dates() -> None:
    """Changing returns beyond the last date used for horizon h leaves h's IC untouched."""
    sig, ret = one_period_panel(n_dates=100, n_assets=20)
    base = ic_by_horizon(sig, ret, horizons=[1, 3])
    ret2 = ret.copy()
    ret2.iloc[-1] = ret2.iloc[-1] * 5 + 0.3  # last row only feeds dates 97..98 at h=3
    changed = ic_by_horizon(sig, ret2, horizons=[1, 3])
    ic_a = ic_series(sig, forward_returns(ret, 3)).iloc[:96]
    ic_b = ic_series(sig, forward_returns(ret2, 3)).iloc[:96]
    pd.testing.assert_series_equal(ic_a, ic_b)
    assert base.table.loc[1, "n_dates"] == changed.table.loc[1, "n_dates"]


def test_deterministic() -> None:
    sig, ret = one_period_panel(n_dates=90, n_assets=20)
    a = ic_by_horizon(sig, ret)
    b = ic_by_horizon(sig, ret)
    pd.testing.assert_frame_equal(a.table, b.table)
    assert a.half_life_periods == b.half_life_periods
    assert a.summary() == b.summary()


def test_result_is_frozen_and_summary_readable() -> None:
    sig, ret = one_period_panel(n_dates=120, n_assets=30)
    res = ic_by_horizon(sig, ret)
    assert isinstance(res, HorizonResult)
    with pytest.raises(AttributeError):
        res.peak_horizon = 3  # type: ignore[misc]
    text = res.summary()
    assert "horizon" in text.lower() and "half-life" in text.lower()
    assert "confidence interval" in text.lower()  # states none is given


# ---- half-life fitting: known answers -----------------------------------------------------


def test_fit_half_life_exact_exponential() -> None:
    h = np.array([1.0, 2.0, 3.0, 6.0, 12.0])
    ic = 0.1 * np.exp(-0.2 * h)
    hl, reason = _fit_half_life(h, ic)
    assert reason is None
    assert hl == pytest.approx(1.0 + math.log(2) / 0.2)


def test_fit_half_life_none_when_not_decreasing() -> None:
    h = np.array([1.0, 2.0, 3.0])
    hl, reason = _fit_half_life(h, np.array([0.01, 0.02, 0.03]))
    assert hl is None and reason is not None and "not decreasing" in reason


def test_fit_half_life_needs_three_positive_points() -> None:
    h = np.array([1.0, 2.0, 3.0, 6.0])
    hl, reason = _fit_half_life(h, np.array([0.05, -0.01, 0.02, -0.03]))
    assert hl is None and reason is not None and "positive" in reason


def test_pure_noise_gives_no_confident_half_life_claim() -> None:
    g = np.random.default_rng(5)
    dates = month_ends(150)
    sig = pd.DataFrame(g.standard_normal((150, 30)), dates)
    ret = pd.DataFrame(0.05 * g.standard_normal((150, 30)), dates)
    res = ic_by_horizon(sig, ret)
    assert (res.table["t_stat"].abs() < 3.5).all()
    # Whatever is returned, it is either None-with-reason or a float without a reason.
    assert (res.half_life_periods is None) == (res.reason is not None)


# ---- edge cases ---------------------------------------------------------------------------


def test_too_little_data_raises() -> None:
    sig, ret = one_period_panel(n_dates=8, n_assets=10)
    with pytest.raises(InsufficientDataError, match="horizon"):
        ic_by_horizon(sig, ret)


def test_horizon_longer_than_sample_raises() -> None:
    sig, ret = one_period_panel(n_dates=30, n_assets=10)
    with pytest.raises(InsufficientDataError, match="horizon 24"):
        ic_by_horizon(sig, ret, horizons=(1, 24))


def test_too_few_assets_raises() -> None:
    sig, ret = one_period_panel(n_dates=60, n_assets=4)
    with pytest.warns(DataDroppedWarning), pytest.raises(InsufficientDataError):
        ic_by_horizon(sig, ret, horizons=(1,))


def test_constant_signal_raises() -> None:
    sig, ret = one_period_panel(n_dates=60, n_assets=10)
    sig[:] = 1.0
    with pytest.warns(DataDroppedWarning), pytest.raises(InsufficientDataError):
        ic_by_horizon(sig, ret, horizons=(1,))


def test_interior_nans_warn_once_and_still_work() -> None:
    sig, ret = one_period_panel(n_dates=120, n_assets=12)
    sig.iloc[40, :] = np.nan  # a whole date missing -> interior NaN IC, dropped with a warning
    with pytest.warns(DataDroppedWarning) as rec:
        res = ic_by_horizon(sig, ret, horizons=(1, 2))
    msgs = [str(w.message) for w in rec if issubclass(w.category, DataDroppedWarning)]
    assert len(msgs) == len(set(msgs))  # duplicates across horizons are collapsed
    assert res.table.loc[1, "n_dates"] == 118


def test_scattered_nans_are_tolerated() -> None:
    sig, ret = one_period_panel(n_dates=120, n_assets=30)
    g = np.random.default_rng(1)
    mask = g.random(sig.shape) < 0.05
    res = ic_by_horizon(sig.mask(mask), ret.mask(g.random(ret.shape) < 0.05), horizons=(1, 3))
    assert res.table.loc[1, "mean_ic"] == pytest.approx(RHO, abs=0.05)


@pytest.mark.parametrize(
    "horizons",
    [(), [], (1, 1, 2), (0, 1), (-1, 2), (1.5, 2), (True, 2), "12", (1, "2"), 3, (1, None)],
)
def test_bad_horizons(horizons: object) -> None:
    sig, ret = one_period_panel(n_dates=60, n_assets=10)
    with pytest.raises(InputError, match="horizons"):
        ic_by_horizon(sig, ret, horizons=horizons)  # type: ignore[arg-type]


def test_bad_other_arguments() -> None:
    sig, ret = one_period_panel(n_dates=60, n_assets=10)
    with pytest.raises(InputError, match="method"):
        ic_by_horizon(sig, ret, method="kendall")  # type: ignore[arg-type]
    with pytest.raises(InputError, match="min_assets"):
        ic_by_horizon(sig, ret, min_assets=1)
    with pytest.raises(InputError, match="hac_lags"):
        ic_by_horizon(sig, ret, hac_lags=-1)
    with pytest.raises(InputError, match="DataFrame"):
        ic_by_horizon(sig.iloc[:, 0], ret)  # type: ignore[arg-type]
    with pytest.raises(InputError, match="below -1"):
        ic_by_horizon(sig, ret - 2.0)


def test_partial_overlap_warns_once() -> None:
    sig, ret = one_period_panel(n_dates=80, n_assets=10)
    with pytest.warns(DataDroppedWarning, match="partly overlap") as rec:
        ic_by_horizon(sig.iloc[:70], ret, horizons=(1, 2, 3))
    assert sum("partly overlap" in str(w.message) for w in rec) == 1


def test_explicit_hac_lags_below_overlap_floor_is_rejected() -> None:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2000-01-31", periods=120, freq="ME")
    signal = pd.DataFrame(rng.standard_normal((120, 30)), index=dates)
    returns = pd.DataFrame(rng.standard_normal((120, 30)) * 0.05, index=dates)
    with pytest.raises(InputError, match="too small"):
        ic_by_horizon(signal, returns, horizons=(12,), hac_lags=0)
    ok = ic_by_horizon(signal, returns, horizons=(12,), hac_lags=11)
    assert ok.table.loc[12, "n_dates"] > 0
