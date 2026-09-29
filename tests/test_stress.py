"""Adversarial and property-style stress tests for the public alphafade API.

Covers:
- Determinism given a seed (50 seeds for fast functions, fewer for slow ones).
- No exceptions on random valid inputs.
- Half-life CI contains the point estimate.
- All scalar outputs finite.
- Edge inputs: very short series, all-NaN, constant, interior-NaN stretch, single column,
  duplicate index, unsorted index, timezone-aware index, mixed frequencies.
"""

from __future__ import annotations

import math
import warnings as _warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

import alphafade as af
from tests.conftest import month_ends

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

_TRUE_TAU = 5.0  # decay timescale in years (true half-life = ln(2)*5 ≈ 3.47 years)


def _monthly_returns(n: int, seed: int, mean: float = 0.005, vol: float = 0.02) -> pd.Series:
    """Random monthly strategy returns of length n."""
    rng = np.random.default_rng(seed)
    return pd.Series(mean + vol * rng.standard_normal(n), index=month_ends(n))


def _decaying_ic(n: int, seed: int, a: float = 0.05, tau: float = _TRUE_TAU) -> pd.Series:
    """Monthly per-date IC with planted exponential decay and Gaussian noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 12.0
    return pd.Series(
        a * np.exp(-t / tau) + rng.normal(0, 0.04, n),
        index=month_ends(n),
    )


def _random_panel(n_dates: int, n_assets: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random (signal, forward-returns) panel with the same dates and assets."""
    rng = np.random.default_rng(seed)
    idx = month_ends(n_dates)
    sig = pd.DataFrame(rng.standard_normal((n_dates, n_assets)), index=idx)
    # Give the signal a small positive predictive relationship to fwd so ICs are real.
    fwd = 0.05 * sig + pd.DataFrame(rng.standard_normal((n_dates, n_assets)), index=idx)
    return sig, fwd


# ─────────────────────────────────────────────────────────────────────────────
# Property: determinism given a seed
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", range(5))
def test_fit_decay_deterministic_given_seed(seed: int) -> None:
    """fit_decay with the same integer seed produces bit-identical results."""
    ic = _decaying_ic(120, seed)
    fit1 = af.fit_decay(ic, n_boot=200, rng=seed)
    fit2 = af.fit_decay(ic, n_boot=200, rng=seed)
    assert fit1.decay_rate == fit2.decay_rate
    assert fit1.rate_ci == fit2.rate_ci
    assert fit1.p_value == fit2.p_value
    assert fit1.initial_level == fit2.initial_level
    np.testing.assert_array_equal(fit1.fitted.to_numpy(), fit2.fitted.to_numpy())


@pytest.mark.parametrize("seed", range(5))
def test_fit_decay_different_seeds_differ(seed: int) -> None:
    """Two different seeds almost always produce different CIs (bootstrap is not fixed)."""
    ic = _decaying_ic(120, seed)
    fit_a = af.fit_decay(ic, n_boot=200, rng=seed)
    fit_b = af.fit_decay(ic, n_boot=200, rng=seed + 1000)
    # The point estimate (no randomness) must be the same.
    assert fit_a.decay_rate == pytest.approx(fit_b.decay_rate, abs=0)
    # The bootstrap CI almost always differs across seeds.
    assert fit_a.rate_ci != fit_b.rate_ci


# ─────────────────────────────────────────────────────────────────────────────
# Property: no exceptions on random valid inputs
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", range(20))
def test_fit_decay_no_exception_on_random_input(seed: int) -> None:
    """fit_decay never raises on a random valid monthly series of 120 observations.

    Random noise without planted decay can push the exponential fit to its bound,
    legitimately emitting FitWarning.  The autouse fixture converts unexpected
    AlphaFadeWarnings to errors, so we explicitly downgrade FitWarning back to
    "always" here to distinguish "raised an exception" from "emitted a warning".
    """
    ic = _monthly_returns(120, seed)
    with _warnings.catch_warnings():
        # Allow FitWarning without making it an error (random data can hit the bound).
        _warnings.filterwarnings("always", category=af.FitWarning)
        fit = af.fit_decay(ic, n_boot=100, rng=seed)
    assert math.isfinite(fit.decay_rate)
    assert 0.0 <= fit.p_value <= 1.0


@pytest.mark.parametrize("seed", range(50))
def test_rolling_sharpe_no_exception_on_random_input(seed: int) -> None:
    """rolling_sharpe never raises; non-NaN outputs are finite."""
    r = _monthly_returns(60, seed)
    sr = af.rolling_sharpe(r, window=12)
    assert sr.dropna().apply(math.isfinite).all()


@pytest.mark.parametrize("seed", range(50))
def test_ic_series_no_exception_and_bounded(seed: int) -> None:
    """ic_series on random data: all non-NaN ICs lie in [-1, 1]."""
    sig, fwd = _random_panel(60, 20, seed)
    ic = af.ic_series(sig, fwd, min_assets=5)
    valid = ic.dropna()
    assert len(valid) > 0
    assert (valid >= -1.0 - 1e-12).all(), f"seed={seed}: min IC = {valid.min()}"
    assert (valid <= 1.0 + 1e-12).all(), f"seed={seed}: max IC = {valid.max()}"


@pytest.mark.parametrize("seed", range(20))
def test_find_break_no_exception_on_random_input(seed: int) -> None:
    """find_break never raises on 120 monthly observations."""
    r = _monthly_returns(120, seed)
    res = af.find_break(r)
    assert 0.0 <= res.p_value <= 1.0
    assert math.isfinite(res.stat)
    assert math.isfinite(res.change)


# ─────────────────────────────────────────────────────────────────────────────
# Property: half-life CI contains the point estimate
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", range(10))
def test_fit_decay_ci_contains_half_life_estimate_when_detected(seed: int) -> None:
    """When decay is detected, [ci_low, ci_high] must bracket half_life_years.

    ci_low = ln(2)/rate_ci[1] and ci_high = ln(2)/rate_ci[0].  For this to hold,
    rate_ci[0] <= decay_rate <= rate_ci[1] must be true.  A percentile bootstrap
    that is well-calibrated satisfies this because resampling from centred residuals
    produces a bootstrap distribution centred on the original estimate.  Systematic
    failure here indicates biased bootstrap CI construction.
    """
    ic = _decaying_ic(240, seed)
    fit = af.fit_decay(ic, n_boot=300, rng=seed)
    if not fit.decay_detected:
        pytest.skip(f"seed={seed}: no decay detected — skip CI containment check")
    assert fit.ci_low is not None and fit.ci_high is not None
    assert fit.ci_low <= fit.half_life_years + 1e-9, (
        f"seed={seed}: half_life={fit.half_life_years:.4f} < ci_low={fit.ci_low:.4f}. "
        f"rate={fit.decay_rate:.4f}, rate_ci={fit.rate_ci}"
    )
    assert fit.half_life_years <= fit.ci_high + 1e-9, (
        f"seed={seed}: half_life={fit.half_life_years:.4f} > ci_high={fit.ci_high:.4f}. "
        f"rate={fit.decay_rate:.4f}, rate_ci={fit.rate_ci}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Property: all scalar outputs are finite
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", range(20))
def test_fit_decay_all_scalar_outputs_finite(seed: int) -> None:
    """Every scalar field of DecayFit is a finite number (no nan or inf)."""
    ic = _decaying_ic(120, seed)
    fit = af.fit_decay(ic, n_boot=100, rng=seed)
    scalars = {
        "decay_rate": fit.decay_rate,
        "p_value": fit.p_value,
        "initial_level": fit.initial_level,
        "r2": fit.r2,
        "r2_exponential": fit.r2_exponential,
        "r2_linear": fit.r2_linear,
        "aic_exponential": fit.aic_exponential,
        "aic_linear": fit.aic_linear,
        "linear_slope": fit.linear_slope,
    }
    for name, val in scalars.items():
        assert math.isfinite(val), f"seed={seed}: {name} = {val!r}"
    assert all(math.isfinite(v) for v in fit.rate_ci), f"seed={seed}: rate_ci = {fit.rate_ci!r}"
    assert fit.fitted.notna().all(), f"seed={seed}: fitted contains NaN"


# ─────────────────────────────────────────────────────────────────────────────
# Edge: very short series
# ─────────────────────────────────────────────────────────────────────────────


def test_fit_decay_fewer_than_20_obs_raises() -> None:
    """fit_decay needs at least 20 non-missing observations."""
    short = pd.Series(np.linspace(0.10, 0.05, 15), index=month_ends(15))
    with pytest.raises(af.InsufficientDataError, match="20"):
        af.fit_decay(short, n_boot=100)


def test_find_break_fewer_than_20_obs_raises() -> None:
    """find_break needs at least 20 observations."""
    short = _monthly_returns(10, 0)
    with pytest.raises(af.InsufficientDataError):
        af.find_break(short)


def test_chow_test_too_few_after_date_raises() -> None:
    """chow_test needs at least 5 observations on each side of the break date."""
    r = _monthly_returns(30, 42)
    # date = 3rd-to-last → only 2 observations after → raises
    late_date = r.index[-3]
    with pytest.raises(af.InsufficientDataError):
        af.chow_test(r, late_date)


def test_rolling_sharpe_window_larger_than_n_raises() -> None:
    """Window larger than the series length raises InsufficientDataError."""
    r = _monthly_returns(10, 0)
    with pytest.raises(af.InsufficientDataError, match="window"):
        af.rolling_sharpe(r, window=50)


def test_publication_gap_too_few_in_sample_raises() -> None:
    """publication_gap requires at least 12 in-sample observations."""
    idx = pd.date_range("1990-01-31", periods=5, freq="ME")
    r = pd.Series([0.01] * 5, index=idx)
    # sample_end is the last date → only 5 in-sample obs → raises
    with pytest.raises(af.InsufficientDataError, match="in-sample"):
        af.publication_gap(r, sample_end="1990-05-31", publication_date="1995-01-01")


# ─────────────────────────────────────────────────────────────────────────────
# Edge: all-NaN input
# ─────────────────────────────────────────────────────────────────────────────


def test_fit_decay_all_nan_raises() -> None:
    """All-NaN series raises InsufficientDataError (no non-missing values)."""
    s = pd.Series([float("nan")] * 60, index=month_ends(60))
    with pytest.raises(af.InsufficientDataError):
        af.fit_decay(s, n_boot=100)


def test_find_break_all_nan_raises() -> None:
    s = pd.Series([float("nan")] * 60, index=month_ends(60))
    with pytest.raises(af.InsufficientDataError):
        af.find_break(s)


def test_rolling_sharpe_all_nan_raises() -> None:
    """An all-NaN series has nothing to compute; say so instead of returning all-NaN."""
    idx = pd.date_range("2000-01-31", periods=60, freq="ME")
    with pytest.raises(af.InsufficientDataError, match="non-missing"):
        af.rolling_sharpe(pd.Series(np.nan, index=idx), window=12)


# ─────────────────────────────────────────────────────────────────────────────
# Edge: interior NaN stretch (DataDroppedWarning expected)
# ─────────────────────────────────────────────────────────────────────────────


def test_fit_decay_interior_nan_warns_and_succeeds() -> None:
    """A gap of NaNs in the middle of the series triggers DataDroppedWarning but still fits."""
    ic = _decaying_ic(120, 0)
    ic_with_gap = ic.copy()
    ic_with_gap.iloc[40:46] = float("nan")  # 6 interior NaN values
    with pytest.warns(af.DataDroppedWarning, match="middle"):
        fit = af.fit_decay(ic_with_gap, n_boot=100, rng=0)
    assert math.isfinite(fit.decay_rate)
    assert fit.n_obs == 114  # 120 - 6


def test_ic_series_partial_panel_overlap_warns() -> None:
    """ic_series with non-identical column sets warns about dropped assets."""
    dates = month_ends(12)
    rng = np.random.default_rng(7)
    sig = pd.DataFrame(rng.standard_normal((12, 5)), index=dates, columns=list("abcde"))
    fwd = pd.DataFrame(rng.standard_normal((12, 5)), index=dates, columns=list("abcfg"))
    with pytest.warns(af.DataDroppedWarning, match="partly overlap"):
        ic = af.ic_series(sig, fwd, min_assets=3)
    # Three shared assets (a, b, c) → IC is defined for most dates.
    assert ic.notna().any()


# ─────────────────────────────────────────────────────────────────────────────
# Edge: constant series
# ─────────────────────────────────────────────────────────────────────────────


def test_fit_decay_constant_raises_input_error() -> None:
    """A constant series has no decay; fit_decay must raise InputError."""
    s = pd.Series([0.05] * 60, index=month_ends(60))
    with pytest.raises(af.InputError, match="constant"):
        af.fit_decay(s, n_boot=100)


def test_find_break_constant_raises_input_error() -> None:
    """A constant series has nothing to test; find_break must raise InputError."""
    s = pd.Series([0.01] * 60, index=month_ends(60))
    with pytest.raises(af.InputError, match="constant"):
        af.find_break(s)


def test_rolling_sharpe_constant_window_gives_nan_not_inf() -> None:
    """Constant returns produce zero volatility; rolling_sharpe must return NaN not infinity.

    A constant window has spread = max - min = 0.  The code guards against dividing by
    zero by setting std to NaN when spread == 0.  Any infinite value here means the guard
    is broken.
    """
    n = 60
    s = pd.Series([0.01] * n, index=month_ends(n))
    sr = af.rolling_sharpe(s, window=12)
    inf_mask = sr.apply(lambda x: math.isinf(x) if not math.isnan(x) else False)
    assert not inf_mask.any(), (
        f"rolling_sharpe returned infinite values on a constant series: {sr[inf_mask]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Edge: bad index (duplicate, unsorted, timezone-aware)
# ─────────────────────────────────────────────────────────────────────────────


def test_duplicate_index_raises_input_error() -> None:
    """A DatetimeIndex with duplicate entries raises InputError."""
    idx = month_ends(60)
    # Insert one extra copy of the 10th date.
    idx_dup = idx.insert(10, idx[10])
    s = pd.Series(np.arange(61, dtype=float), index=idx_dup)
    with pytest.raises(af.InputError, match="duplicate"):
        af.fit_decay(s, n_boot=100)


def test_unsorted_index_raises_input_error() -> None:
    """A reversed DatetimeIndex raises InputError."""
    idx = month_ends(60)
    s = pd.Series(np.arange(60, dtype=float), index=idx[::-1])
    with pytest.raises(af.InputError, match="sorted"):
        af.fit_decay(s, n_boot=100)


def test_timezone_aware_index_raises_input_error() -> None:
    """A timezone-aware DatetimeIndex raises InputError for every entry point."""
    idx = pd.date_range("2000-01-31", periods=60, freq="ME", tz="UTC")
    s = pd.Series(np.random.default_rng(0).normal(0.005, 0.02, 60), index=idx)
    with pytest.raises(af.InputError, match="timezone"):
        af.rolling_sharpe(s, window=12)


def test_timezone_aware_index_raises_for_ic_series() -> None:
    idx = pd.date_range("2000-01-31", periods=12, freq="ME", tz="Europe/London")
    rng = np.random.default_rng(0)
    sig = pd.DataFrame(rng.standard_normal((12, 5)), index=idx)
    fwd = pd.DataFrame(rng.standard_normal((12, 5)), index=idx)
    with pytest.raises(af.InputError, match="timezone"):
        af.ic_series(sig, fwd, min_assets=3)


# ─────────────────────────────────────────────────────────────────────────────
# Edge: mixed frequencies
# ─────────────────────────────────────────────────────────────────────────────


def test_mixed_frequency_rolling_sharpe_raises_frequency_error() -> None:
    """A series containing both monthly and daily dates raises FrequencyError.

    Construction: 12 monthly dates (1 year) then 30 consecutive daily dates appended
    immediately after.  The monthly gaps cover ~360 days; the 29 daily gaps cover 29 days.
    With freq=None, infer_freq detects the mixture and raises.
    """
    monthly = pd.date_range("2000-01-31", periods=12, freq="ME")
    daily = pd.date_range("2001-01-31", periods=30, freq="D")
    idx = monthly.append(daily)
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0.005, 0.02, len(idx)), index=idx)
    with pytest.raises(af.FrequencyError):
        af.rolling_sharpe(s, window=12)


# ─────────────────────────────────────────────────────────────────────────────
# Edge: single column / minimal panel
# ─────────────────────────────────────────────────────────────────────────────


def test_forward_returns_single_column_dataframe_works() -> None:
    """forward_returns on a one-column DataFrame returns a DataFrame of the same shape."""
    idx = month_ends(10)
    ret = pd.DataFrame(
        {"strategy": [0.01, 0.02, -0.01, 0.03, 0.01, 0.02, 0.0, -0.02, 0.01, 0.005]},
        index=idx,
    )
    fwd = af.forward_returns(ret, periods=1)
    assert isinstance(fwd, pd.DataFrame)
    assert fwd.shape == ret.shape
    # The last row must be NaN (no future return known).
    assert math.isnan(fwd.iloc[-1, 0])
    # All other rows should be finite.
    assert fwd.iloc[:-1].notna().all().all()


def test_ic_series_single_asset_all_nan_and_warns() -> None:
    """With only 1 asset in the panel, every IC is NaN and DataDroppedWarning is emitted."""
    dates = month_ends(12)
    rng = np.random.default_rng(0)
    sig = pd.DataFrame({"stock_a": rng.standard_normal(12)}, index=dates)
    fwd = pd.DataFrame({"stock_a": rng.standard_normal(12)}, index=dates)
    # min_assets=2 (minimum allowed) but only 1 asset → all NaN
    with pytest.warns(af.DataDroppedWarning, match="fewer than 2"):
        ic = af.ic_series(sig, fwd, min_assets=2)
    assert ic.isna().all()


# ─────────────────────────────────────────────────────────────────────────────
# Edge: publication_gap boundary conditions
# ─────────────────────────────────────────────────────────────────────────────


def test_publication_gap_pub_before_sample_end_raises() -> None:
    """publication_gap must raise if publication_date <= sample_end."""
    idx = pd.date_range("1980-01-31", periods=240, freq="ME")
    r = pd.Series([0.01] * 240, index=idx)
    with pytest.raises(af.InputError, match="publication_date"):
        af.publication_gap(r, sample_end="1995-12-31", publication_date="1990-01-01")


def test_publication_gap_data_ends_before_pub_date_raises() -> None:
    """publication_gap must raise if no observations reach the publication date."""
    idx = pd.date_range("1980-01-31", periods=120, freq="ME")
    r = pd.Series(np.random.default_rng(0).normal(0.01, 0.02, 120), index=idx)
    # Data ends 1989-12-31; publication_date 1995-01-01 → no post-pub obs
    with pytest.raises(af.InsufficientDataError):
        af.publication_gap(r, sample_end="1985-12-31", publication_date="1995-01-01")


# ─────────────────────────────────────────────────────────────────────────────
# Edge: crowding_score NaN membership warns
# ─────────────────────────────────────────────────────────────────────────────


def test_crowding_score_nan_membership_warns() -> None:
    """NaN values in the membership DataFrame emit DataDroppedWarning (not silently ignored).

    The short leg is all-False (no members) so the only expected DataDroppedWarning
    is the one from _membership about the NaN in long_members.  Nested pytest.warns
    contexts compete for warnings (the inner one captures them before the outer can
    check its criterion), so we use a single context and inspect the recorded list.
    """
    rng = np.random.default_rng(42)
    return_dates = pd.date_range("2000-01-07", periods=80, freq="W-FRI")
    formation_dates = return_dates[51::4]

    rets = pd.DataFrame(
        rng.normal(0, 0.02, (80, 20)),
        index=return_dates,
    )
    # Build membership with one deliberate NaN entry, using object dtype so pandas
    # detects it with .isna().
    long_m = pd.DataFrame(True, index=formation_dates, columns=rets.columns).astype(object)
    long_m.iloc[0, 0] = np.nan
    # All-False short leg → no short-leg thin-stocks or thin-legs warnings.
    short_m = pd.DataFrame(False, index=formation_dates, columns=rets.columns)

    with pytest.warns(af.DataDroppedWarning, match="missing") as warning_list:
        af.crowding_score(rets, long_m, short_m, window=52)
    assert any("missing" in str(w.message) for w in warning_list)


# ─────────────────────────────────────────────────────────────────────────────
# Edge: forward_returns rejects values below -1
# ─────────────────────────────────────────────────────────────────────────────


def test_forward_returns_rejects_below_minus_one() -> None:
    """A return of -1.5 (a loss > 100%) is rejected as probable unit error."""
    idx = month_ends(10)
    s = pd.Series([0.01, 0.02, -1.5, 0.03, 0.01, 0.02, 0.0, -0.02, 0.01, 0.0], index=idx)
    with pytest.raises(af.InputError, match="below -1"):
        af.forward_returns(s, periods=1)


# ─────────────────────────────────────────────────────────────────────────────
# Edge: fit_decay on a rolling series adds a note and widens blocks
# ─────────────────────────────────────────────────────────────────────────────


def test_fit_decay_on_rolling_ic_adds_overlap_note() -> None:
    """When fed a rolling_ic output (which carries WINDOW_ATTR), fit_decay notes the overlap."""
    sig, fwd = _random_panel(120, 10, seed=0)
    ric = af.rolling_ic(sig, fwd, window=12)
    fit = af.fit_decay(ric, n_boot=100, rng=0)
    assert any("rolling" in note.lower() or "overlap" in note.lower() for note in fit.notes), (
        f"Expected an overlap/rolling note; got notes={fit.notes!r}"
    )
    # Block size must be at least the rolling window.
    assert fit.block_size >= 12


# ─────────────────────────────────────────────────────────────────────────────
# Edge: ic_series output name is always "ic"
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", range(5))
def test_ic_series_output_name(seed: int) -> None:
    """ic_series always returns a Series named 'ic'."""
    sig, fwd = _random_panel(30, 10, seed)
    ic = af.ic_series(sig, fwd, min_assets=5)
    assert ic.name == "ic"


# ─────────────────────────────────────────────────────────────────────────────
# Edge: rolling_sharpe output name and attrs
# ─────────────────────────────────────────────────────────────────────────────


def test_rolling_sharpe_carries_window_attr() -> None:
    """rolling_sharpe attaches the window length to attrs for downstream use."""
    r = _monthly_returns(60, 0)
    sr = af.rolling_sharpe(r, window=24)
    from alphafade.rolling import WINDOW_ATTR

    assert sr.attrs.get(WINDOW_ATTR) == 24
    assert sr.name == "rolling_sharpe"


# ---------------------------------------------------------------------------------------
# Silent-garbage inputs that must be rejected up front


@pytest.mark.parametrize(
    "call",
    [
        lambda s: af.fit_decay(s, n_boot=100, rng=0),
        lambda s: af.find_break(s),
        lambda s: af.rolling_sharpe(s, window=12),
        lambda s: af.analyze(s, n_boot=100, rng=0),
    ],
    ids=["fit_decay", "find_break", "rolling_sharpe", "analyze"],
)
def test_boolean_and_overflow_scale_series_are_rejected(call: Any) -> None:
    idx = pd.date_range("2000-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(0)
    with pytest.raises(af.InputError, match="True/False"):
        call(pd.Series([True, False] * 60, index=idx))
    with pytest.raises(af.InputError, match="overflow"):
        call(pd.Series(1e300 * rng.normal(size=120), index=idx))


def test_boolean_panel_is_rejected() -> None:
    idx = pd.date_range("2000-01-31", periods=30, freq="ME")
    flags = pd.DataFrame(np.random.default_rng(0).random((30, 10)) > 0.5, index=idx)
    with pytest.raises(af.InputError, match="True/False"):
        af.forward_returns(flags)
