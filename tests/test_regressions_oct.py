"""Regressions for bugs found by the independent verification pass (2026-10-04)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import alphafade as af

IDX = pd.date_range("2000-01-31", periods=60, freq="ME")


def test_constant_excess_return_with_varying_rf_is_nan_not_huge() -> None:
    rf = pd.Series(np.linspace(0, 0.005, 60), index=IDX)
    r = 0.01 + rf  # excess return is exactly 0.01 every month: zero volatility
    out = af.rolling_sharpe(r, window=12, rf=rf)
    assert out.dropna().empty


@pytest.mark.parametrize("rf", [np.zeros(60), "0.001", None, pd.DataFrame({"a": np.zeros(60)})])
def test_bad_rf_types_raise_input_error(rf: object) -> None:
    r = pd.Series(np.random.default_rng(0).normal(0.01, 0.02, 60), index=IDX)
    with pytest.raises(af.InputError, match="rf"):
        af.rolling_sharpe(r, window=12, rf=rf)  # type: ignore[arg-type]


def test_nan_in_rf_series_warns() -> None:
    r = pd.Series(np.random.default_rng(0).normal(0.01, 0.02, 60), index=IDX)
    rf = pd.Series(0.001, index=IDX)
    rf.iloc[30] = np.nan
    with pytest.warns(af.DataDroppedWarning, match="rf"):
        af.rolling_sharpe(r, window=12, rf=rf)


def test_rolling_ic_with_no_usable_ic_raises() -> None:
    sig = pd.DataFrame(np.nan, index=IDX, columns=range(10))
    fwd = pd.DataFrame(np.random.default_rng(0).normal(size=(60, 10)), index=IDX)
    with pytest.raises(af.InsufficientDataError, match="usable IC"):
        af.rolling_ic(sig, fwd, window=12)


def test_min_periods_longer_than_window_has_clear_message() -> None:
    sig = pd.DataFrame(np.random.default_rng(0).normal(size=(60, 10)), index=IDX)
    fwd = pd.DataFrame(np.random.default_rng(1).normal(size=(60, 10)), index=IDX)
    with pytest.raises(af.InputError, match="min_periods=20 is larger than the window"):
        af.rolling_ic(sig, fwd, window=12, min_periods=20)


def test_crowding_formation_dates_after_last_return_are_nan_and_warn() -> None:
    rng = np.random.default_rng(0)
    weeks = pd.date_range("1990-01-05", periods=200, freq="W-FRI")
    rets = pd.DataFrame(rng.normal(0, 0.02, (200, 20)), index=weeks)
    form = pd.date_range("1991-01-31", "1995-12-31", freq="ME")  # returns end in late 1993
    long_ = pd.DataFrame(np.tile(np.arange(20) < 10, (len(form), 1)), index=form)
    with pytest.warns(af.DataDroppedWarning, match="after the last stock return"):
        out = af.crowding_score(rets, long_, ~long_)
    assert out.loc[out.index > weeks[-1] + pd.Timedelta(days=14), "mean"].isna().all()
    assert out.loc[out.index <= weeks[-1], "mean"].notna().any()


def test_verdict_says_higher_when_returns_rose_after_publication() -> None:
    idx = pd.date_range("1980-01-31", periods=480, freq="ME")
    rng = np.random.default_rng(0)
    mean = np.where(idx < pd.Timestamp("1995-01-01"), 0.002, 0.012)
    r = pd.Series(mean + rng.normal(0, 0.01, 480), index=idx)
    rep = af.analyze(r, sample_end="1990-12-31", publication_date="1995-01-01", n_boot=200, rng=0)
    assert rep.publication is not None and rep.publication.post_publication_decline < 0
    text = rep.verdict()
    assert "higher after publication" in text
    assert "-" not in text.split("Returns are")[1].split("%")[0]


def test_compare_signals_results_do_not_depend_on_column_order() -> None:
    rng = np.random.default_rng(1)
    idx = pd.date_range("1990-01-31", periods=240, freq="ME")
    df = pd.DataFrame(rng.normal(0.003, 0.02, (240, 5)), index=idx, columns=list("abcde"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", af.FitWarning)
        a = af.compare_signals(df, n_boot=200, rng=7).table.sort_index()
        b = af.compare_signals(df[list("edcba")], n_boot=200, rng=7).table.sort_index()
        c = af.compare_signals(df[list("ace")], n_boot=200, rng=7).table.sort_index()
    pd.testing.assert_series_equal(a["p_value"], b["p_value"])
    pd.testing.assert_series_equal(a.loc[list("ace"), "p_value"], c["p_value"])


def test_walk_forward_never_says_trust_when_full_sample_has_no_decay() -> None:
    idx = pd.date_range("1970-01-31", periods=480, freq="ME")
    t = np.arange(480) / 12
    y = np.where(t < 12, 0.1 * np.exp(-t / 3), 0.1) + np.random.default_rng(0).normal(0, 0.01, 480)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", af.FitWarning)
        res = af.walk_forward_decay(
            pd.Series(y, index=idx), min_obs=60, step=24, n_boot=100, rng=0
        )
    assert not res.table["decay_detected"].iloc[-1]  # the setup the test is about
    assert res.table["decay_detected"].any()
    assert "supports trusting" not in res.summary()
    assert "no full-sample half-life" in res.summary()


def _horizon_panels(sign: float, strength: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(2)
    idx = pd.date_range("1990-01-31", periods=200, freq="ME")
    sig = pd.DataFrame(rng.standard_normal((200, 40)), index=idx)
    ret = 0.05 * (sign * strength * sig.shift(1).fillna(0) + rng.standard_normal((200, 40)))
    return sig, ret


def test_horizon_summary_flags_a_reversed_signal() -> None:
    sig, ret = _horizon_panels(-1.0, 0.4)
    res = af.ic_by_horizon(sig, ret, horizons=(1, 2, 3, 6))
    assert "OPPOSITE" in res.summary()
    assert "predicts best" not in res.summary()
    assert res.half_life_periods is None


def test_horizon_summary_flags_noise_and_gives_no_half_life() -> None:
    sig, ret = _horizon_panels(1.0, 0.0)
    res = af.ic_by_horizon(sig, ret, horizons=(1, 2, 3, 6))
    assert (res.table["t_stat"].abs() < 2).all()  # the setup the test is about
    assert "may be noise" in res.summary()
    assert res.half_life_periods is None


def test_signal_lifetime_accepts_numpy_scalars() -> None:
    idx = pd.date_range("1970-01-31", periods=480, freq="ME")
    t = np.arange(480) / 12
    y = 0.1 * np.exp(-t / 5) + np.random.default_rng(0).normal(0, 0.01, 480)
    fit = af.fit_decay(pd.Series(y, index=idx), n_boot=200, rng=0)
    a = af.signal_lifetime(fit, fraction=np.float32(0.5))
    b = af.signal_lifetime(fit, fraction=0.5)
    assert a.years_from_start == pytest.approx(b.years_from_start, rel=1e-6)
