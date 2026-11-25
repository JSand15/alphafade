"""fit_floor_decay: known-answer synthetic data, honesty checks, and error handling."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import optimize

import alphafade as af
from tests.conftest import month_ends


def curve(
    start: float,
    floor: float,
    half_life: float,
    *,
    years: int = 40,
    sd: float = 0.0,
    seed: int = 0,
) -> pd.Series:
    """Monthly series floor + (start - floor) * 0.5 ** (t / half_life) plus noise."""
    n = years * 12
    t = np.arange(n) / 12
    noise = np.random.default_rng(seed).normal(0, sd, n) if sd else np.zeros(n)
    y = floor + (start - floor) * np.exp(-t * math.log(2) / half_life) + noise
    return pd.Series(y, index=month_ends(n))


# ----------------------------------------------------------------------------- recovery
def test_noise_free_exact_recovery() -> None:
    fit = af.fit_floor_decay(curve(0.10, 0.04, 4.0), n_boot=100, rng=0)
    assert fit.decay_detected
    assert fit.floor == pytest.approx(0.04, abs=2e-5)
    assert fit.start_level == pytest.approx(0.10, abs=5e-5)
    # Month-ends are not exactly 1/12 year apart, so allow a hair of slack.
    assert fit.half_life_years == pytest.approx(4.0, rel=3e-3)
    assert fit.share_remaining == pytest.approx(0.4, abs=1e-3)
    assert fit.floor_outcome == "survives"
    assert fit.r2 == pytest.approx(1.0, abs=1e-5)


def test_positive_floor_survives_and_interval_covers_truth() -> None:
    fit = af.fit_floor_decay(curve(0.10, 0.04, 4.0, years=60, sd=0.02, seed=1), rng=1)
    assert fit.floor_outcome == "survives"
    assert fit.floor_ci is not None
    assert fit.floor_ci[0] < 0.04 < fit.floor_ci[1]
    assert fit.ci_low is not None
    assert fit.ci_high is not None
    assert fit.ci_low < 4.0 < fit.ci_high
    assert "survives" in fit.summary()


def test_zero_floor_is_not_claimed_to_survive() -> None:
    fit = af.fit_floor_decay(curve(0.10, 0.0, 4.0, years=60, sd=0.02, seed=2), rng=2)
    assert fit.decay_detected
    assert fit.floor_outcome == "may_reach_zero"
    assert fit.floor_ci is not None
    assert fit.floor_ci[0] <= 0 <= fit.floor_ci[1]
    assert "all the way to zero" in fit.summary()


def test_edge_that_reverses() -> None:
    fit = af.fit_floor_decay(curve(0.10, -0.03, 4.0, years=60, sd=0.01, seed=3), rng=3)
    assert fit.floor_outcome == "reverses"
    assert fit.floor == pytest.approx(-0.03, abs=0.005)
    assert "opposite sign" in fit.summary()


def test_negative_edge_shrinking_to_a_negative_floor_survives() -> None:
    fit = af.fit_floor_decay(curve(-0.10, -0.02, 4.0, years=60, sd=0.01, seed=4), rng=4)
    assert fit.decay_detected
    assert fit.floor_outcome == "survives"
    assert fit.floor == pytest.approx(-0.02, abs=0.005)
    assert fit.share_remaining == pytest.approx(0.2, abs=0.05)


def test_uses_calendar_time_not_row_counts() -> None:
    monthly = curve(0.10, 0.04, 4.0)
    quarterly = monthly.iloc[2::3]
    fit = af.fit_floor_decay(quarterly, n_boot=100, rng=0)
    assert fit.half_life_years == pytest.approx(4.0, rel=5e-3)
    assert fit.floor == pytest.approx(0.04, abs=2e-4)


def test_fitted_curve_and_model_comparison() -> None:
    perf = curve(0.10, 0.04, 4.0, years=60, sd=0.01, seed=5)
    fit = af.fit_floor_decay(perf, n_boot=100, rng=5)
    assert fit.fitted.index.equals(perf.index)
    # A real floor: the floor model must beat both the zero-floor and the constant model.
    assert fit.aic_floor < fit.aic_exponential
    assert fit.aic_floor < fit.aic_constant


@pytest.mark.parametrize("seed", range(8))
def test_least_squares_fit_matches_scipy(seed: int) -> None:
    """Independent oracle: scipy's general optimiser can't find a better curve than ours."""
    draw = np.random.default_rng(seed)
    start, floor, half = draw.uniform(0.05, 0.2), draw.uniform(-0.03, 0.04), draw.uniform(1, 8)
    perf = curve(start, floor, half, sd=0.02, seed=seed)
    fit = af.fit_floor_decay(perf, n_boot=100, rng=0)
    t = (perf.index - perf.index[0]).days.to_numpy() / 365.25
    y = perf.to_numpy()
    lowest, highest = math.log(2) / t[-1], math.log(2) / max(t[-1] / 100, 0.25)
    solved = optimize.least_squares(
        lambda p: p[0] + p[1] * np.exp(-p[2] * t) - y,
        x0=[floor, start - floor, math.log(2) / half],
        bounds=([-np.inf, -np.inf, lowest], [np.inf, np.inf, highest]),
        xtol=1e-14,
        ftol=1e-14,
    )
    ours = float(((fit.fitted.to_numpy() - y) ** 2).sum())
    theirs = float(2 * solved.cost)
    assert ours <= theirs * (1 + 1e-7)
    if fit.floor is not None:
        assert fit.floor == pytest.approx(solved.x[0], abs=1e-4)
        assert fit.decay_rate == pytest.approx(solved.x[2], rel=1e-3)


# ----------------------------------------------------------------------------- honesty
def assert_no_floor_claim(fit: af.FloorFit) -> None:
    assert not fit.decay_detected
    assert fit.floor is None
    assert fit.floor_ci is None
    assert fit.share_remaining is None
    assert fit.half_life_years is None
    assert fit.ci_low is None
    assert fit.ci_high is None
    assert fit.floor_outcome == "unknown"


def test_constant_series_has_no_detectable_change() -> None:
    rng = np.random.default_rng(11)
    perf = pd.Series(0.05 + rng.normal(0, 0.05, 480), index=month_ends(480))
    fit = af.fit_floor_decay(perf, rng=11)
    assert not fit.change_detected
    assert fit.p_value > 0.05
    assert_no_floor_claim(fit)
    assert "No detectable change" in fit.summary()


def test_straight_line_decline_has_no_identifiable_floor() -> None:
    n = 480
    t = np.arange(n) / 12
    noise = np.random.default_rng(6).normal(0, 0.01, n)
    fit = af.fit_floor_decay(pd.Series(0.10 - 0.002 * t + noise, index=month_ends(n)), rng=6)
    assert fit.change_detected
    assert not fit.floor_identified
    assert_no_floor_claim(fit)
    assert "can't pin down a floor" in fit.summary()


@pytest.mark.parametrize("half_life", [40.0, 80.0])
def test_decay_slower_than_the_sample_gives_no_floor(half_life: float) -> None:
    fit = af.fit_floor_decay(curve(0.10, 0.0, half_life, sd=0.01, seed=7), n_boot=300, rng=7)
    assert not fit.floor_identified
    assert_no_floor_claim(fit)


def test_rising_edge_is_not_decay() -> None:
    fit = af.fit_floor_decay(curve(0.01, 0.08, 5.0, sd=0.01, seed=8), rng=8)
    assert fit.change_detected
    assert fit.floor_identified
    assert_no_floor_claim(fit)
    assert "isn't shrinking" in fit.summary()


def _rates(make: object, seeds: int) -> list[af.FloorFit]:
    return [
        af.fit_floor_decay(make(s), n_boot=200, rng=s)  # type: ignore[operator]
        for s in range(seeds)
    ]


def test_false_alarms_stay_rare_when_early_data_are_noisier() -> None:
    # Constant average, but the first 8 years are 4x as volatile. A bootstrap that shuffles
    # residuals across time calls this "decay" about a third of the time; ours must not.
    def make(seed: int) -> pd.Series:
        sd = np.where(np.arange(480) < 96, 0.12, 0.03)
        noise = np.random.default_rng(seed).standard_normal(480) * sd
        return pd.Series(0.05 + noise, index=month_ends(480))

    fits = _rates(make, 60)
    assert sum(f.change_detected for f in fits) <= 8  # nominal 5% of 60 = 3
    assert sum(f.decay_detected for f in fits) <= 5


def test_zero_floor_is_rarely_called_surviving() -> None:
    fits = _rates(lambda s: curve(0.10, 0.0, 4.0, years=60, sd=0.03, seed=s), 60)
    assert sum(f.floor_outcome == "survives" for f in fits) <= 6  # nominal 2.5% of 60


def test_floor_interval_coverage() -> None:
    fits = _rates(lambda s: curve(0.10, 0.04, 4.0, years=60, sd=0.03, seed=s), 60)
    covered = sum(f.floor_ci is not None and f.floor_ci[0] <= 0.04 <= f.floor_ci[1] for f in fits)
    assert covered >= 50  # nominal 95% of 60 = 57


# ----------------------------------------------------------------------------- behaviour
def test_seed_makes_results_reproducible() -> None:
    perf = curve(0.10, 0.04, 4.0, sd=0.03, seed=9)
    a = af.fit_floor_decay(perf, n_boot=200, rng=3)
    b = af.fit_floor_decay(perf, n_boot=200, rng=3)
    c = af.fit_floor_decay(perf, n_boot=200, rng=np.random.default_rng(3))
    assert a.floor_ci == b.floor_ci == c.floor_ci
    assert a.p_value == b.p_value


@settings(max_examples=10, deadline=None)
@given(scale=st.floats(min_value=1e-3, max_value=1e3), seed=st.integers(0, 1000))
def test_rescaling_the_data_rescales_levels_only(scale: float, seed: int) -> None:
    perf = curve(0.10, 0.04, 4.0, sd=0.01, seed=seed)
    a = af.fit_floor_decay(perf, n_boot=100, rng=1)
    b = af.fit_floor_decay(perf * scale, n_boot=100, rng=1)
    assert a.floor_outcome == b.floor_outcome
    assert a.p_value == pytest.approx(b.p_value)
    if a.floor is not None and b.floor is not None:
        assert b.floor == pytest.approx(a.floor * scale, rel=1e-5, abs=1e-9 * scale)
        assert b.half_life_years == pytest.approx(a.half_life_years, rel=1e-5)


def test_p_value_is_never_zero() -> None:
    fit = af.fit_floor_decay(curve(0.10, 0.0, 4.0, sd=0.001), n_boot=100, rng=0)
    assert fit.p_value == pytest.approx(1 / 101)


def test_interior_missing_values_warn() -> None:
    perf = curve(0.10, 0.04, 4.0, sd=0.01)
    perf.iloc[[50, 51]] = np.nan
    with pytest.warns(af.DataDroppedWarning, match="2 missing"):
        fit = af.fit_floor_decay(perf, n_boot=100, rng=0)
    assert fit.n_obs == 478


def test_rolling_input_widens_blocks_and_adds_note() -> None:
    rng = np.random.default_rng(0)
    idx = month_ends(480)
    returns = pd.Series(0.01 + rng.normal(0, 0.02, 480), index=idx)
    rolled = af.rolling_sharpe(returns, window=36)
    fit = af.fit_floor_decay(rolled, n_boot=100, rng=0)
    assert fit.block_size == 36
    assert any("rolling series" in note for note in fit.notes)


def test_to_dict_is_strict_json() -> None:
    perf = curve(0.10, 0.04, 4.0, sd=0.01)
    fit = af.fit_floor_decay(perf, n_boot=100, rng=0)
    data = fit.to_dict()
    json.dumps(data, allow_nan=False)
    assert data["floor_outcome"] == "survives"
    assert "fitted" not in data
    assert len(fit.to_dict(include_series=True)["fitted"]) == len(perf)
    empty = af.fit_floor_decay(
        pd.Series(np.random.default_rng(1).normal(0, 1, 120), index=month_ends(120)),
        n_boot=100,
        rng=1,
    )
    json.dumps(empty.to_dict(), allow_nan=False)


# ----------------------------------------------------------------------------- errors
def test_too_few_observations() -> None:
    with pytest.raises(af.InsufficientDataError, match="at least 36"):
        af.fit_floor_decay(curve(0.1, 0.0, 1.0, years=2))


def test_constant_input() -> None:
    with pytest.raises(af.InputError, match="constant"):
        af.fit_floor_decay(pd.Series(0.05, index=month_ends(60)))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_boot": 50}, "n_boot"),
        ({"n_boot": 200.0}, "n_boot"),
        ({"n_boot": True}, "n_boot"),
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": 0.6}, "alpha"),
        ({"block_size": 0}, "block_size"),
        ({"block_size": 61}, "block_size"),
        ({"block_size": 2.5}, "block_size"),
        ({"rng": "x"}, "rng"),
    ],
)
def test_bad_arguments(kwargs: dict[str, object], match: str) -> None:
    perf = curve(0.1, 0.0, 1.0, years=5, sd=0.01)
    with pytest.raises(af.InputError, match=match):
        af.fit_floor_decay(perf, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("scale", [1e-200, 1e-300])
def test_vanishingly_small_values_are_refused_cleanly(scale: float) -> None:
    perf = curve(0.1, 0.04, 4.0, years=5, sd=0.01) * scale
    with pytest.raises(af.InputError, match="units"):
        af.fit_floor_decay(perf, n_boot=100)


def test_alpha_too_small_for_n_boot() -> None:
    # With 100 resamples the smallest possible p-value is 1/101, so alpha=0.001 could never
    # detect anything. Say so instead of silently reporting "no change".
    perf = curve(0.1, 0.0, 1.0, years=5, sd=0.01)
    with pytest.raises(af.InputError, match="n_boot"):
        af.fit_floor_decay(perf, n_boot=100, alpha=0.001)
