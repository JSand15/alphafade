from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import alphafade as af
from tests.conftest import month_ends

TRUE_HALF_LIFE = 5 * math.log(2)


def decaying_ic(
    n_years: int, noise_sd: float, seed: int, a: float = 0.10, tau: float = 5.0
) -> pd.Series:
    rng = np.random.default_rng(seed)
    n = n_years * 12
    t = np.arange(n) / 12
    return pd.Series(a * np.exp(-t / tau) + rng.normal(0, noise_sd, n), index=month_ends(n))


def test_noise_free_exact_recovery() -> None:
    fit = af.fit_decay(decaying_ic(40, 0.0, 0), n_boot=100, rng=0)
    # Month-ends are not exactly 1/12 year apart, so allow a hair of slack.
    assert fit.half_life_years == pytest.approx(TRUE_HALF_LIFE, rel=2e-3)
    assert fit.initial_level == pytest.approx(0.10, rel=1e-3)
    assert fit.model == "exponential"
    assert fit.r2 == pytest.approx(1.0, abs=1e-5)


def test_spec_recovers_half_life_within_10pct_on_long_sample() -> None:
    # 80 years of monthly IC with per-date noise sd 0.005 (a cross-section of ~40k assets,
    # or equivalently a long, precise sample).
    fit = af.fit_decay(decaying_ic(80, 0.005, 7), n_boot=300, rng=7)
    assert fit.decay_detected
    assert fit.half_life_years == pytest.approx(TRUE_HALF_LIFE, rel=0.10)
    assert fit.ci_low < TRUE_HALF_LIFE < fit.ci_high


def test_estimator_is_unbiased_across_seeds() -> None:
    # Realistic noise (~3000 assets): single estimates scatter ~6%, but the median is on target.
    errs = []
    for seed in range(25):
        fit = af.fit_decay(decaying_ic(60, 1 / math.sqrt(3000), seed), n_boot=100, rng=seed)
        errs.append(fit.half_life_years / TRUE_HALF_LIFE - 1)
    assert abs(float(np.median(errs))) < 0.03


def test_end_to_end_from_asset_panel() -> None:
    rng = np.random.default_rng(3)
    n, k = 600, 800
    t = np.arange(n) / 12
    rho = 0.10 * np.exp(-t / 5)
    sig = rng.standard_normal((n, k))
    fwd = rho[:, None] * sig + np.sqrt(1 - rho[:, None] ** 2) * rng.standard_normal((n, k))
    idx = month_ends(n)
    ic = af.ic_series(pd.DataFrame(sig, index=idx), pd.DataFrame(fwd, index=idx))
    fit = af.fit_decay(ic, n_boot=300, rng=3)
    assert fit.decay_detected
    assert fit.half_life_years == pytest.approx(TRUE_HALF_LIFE, rel=0.35)


def test_constant_ic_no_significant_decay() -> None:
    rng = np.random.default_rng(11)
    ic = pd.Series(0.05 + rng.normal(0, 0.05, 480), index=month_ends(480))
    fit = af.fit_decay(ic, n_boot=500, rng=11)
    assert not fit.decay_detected
    assert fit.half_life_years is None
    assert fit.ci_low is None
    assert fit.ci_high is None
    assert fit.rate_ci[0] <= 0 <= fit.rate_ci[1]
    assert fit.p_value > 0.025
    assert "No detectable decay" in fit.summary()


def test_growing_edge_is_not_decay() -> None:
    rng = np.random.default_rng(5)
    n = 480
    t = np.arange(n) / 12
    ic = pd.Series(0.02 * np.exp(t / 10) + rng.normal(0, 0.005, n), index=month_ends(n))
    fit = af.fit_decay(ic, n_boot=300, rng=5)
    assert not fit.decay_detected
    assert fit.decay_rate < 0
    assert "growing" in fit.summary()


def test_negative_ic_signal_decays_in_magnitude() -> None:
    ic = -decaying_ic(60, 0.005, 2)
    fit = af.fit_decay(ic, n_boot=300, rng=2)
    assert fit.decay_detected
    assert fit.initial_level < 0
    assert fit.half_life_years == pytest.approx(TRUE_HALF_LIFE, rel=0.10)


def test_too_fast_decay_falls_back_to_linear_with_warning() -> None:
    y = np.zeros(480)
    y[:2] = 1.0
    y += np.random.default_rng(0).normal(0, 0.01, 480)
    with pytest.warns(af.FitWarning, match="linear model is used"):
        fit = af.fit_decay(pd.Series(y, index=month_ends(480)), n_boot=200, rng=0)
    assert fit.model == "linear"
    assert any("failed" in note for note in fit.notes)
    assert fit.fitted.name == "linear_fit"


def test_linear_fallback_on_steep_growth_is_not_decay() -> None:
    # A jump from ~0 to 1 at the very end is growth steeper than the grid allows. The linear
    # line starts at ~0, so a "decay relative to the starting level" must not be claimed.
    n = 480
    t = np.arange(n) / 12
    y = np.where(t < 39.5, 0.0, 1.0) + np.random.default_rng(1).normal(0, 1e-3, n)
    with pytest.warns(af.FitWarning, match="steeper growth"):
        fit = af.fit_decay(pd.Series(y, index=month_ends(n)), n_boot=200, rng=1)
    assert fit.model == "linear"
    assert not fit.decay_detected
    assert any("distinguishable from zero" in note for note in fit.notes)


def test_linear_fallback_detects_decay() -> None:
    n = 480
    y = np.zeros(n)
    y[:2] = 20.0  # a spike only an ultra-fast exponential could fit -> bound -> linear
    y += np.linspace(1.0, 0.2, n) + np.random.default_rng(4).normal(0, 0.01, n)
    with pytest.warns(af.FitWarning):
        fit = af.fit_decay(pd.Series(y, index=month_ends(n)), n_boot=200, rng=4)
    assert fit.model == "linear"
    assert fit.decay_detected
    assert fit.half_life_years is not None
    assert fit.half_life_years > 0
    assert "falling by" in fit.summary()


def test_linear_trend_prefers_linear_model() -> None:
    rng = np.random.default_rng(9)
    n = 600
    t = np.arange(n) / 12
    y = pd.Series(0.10 - 0.002 * t + rng.normal(0, 0.002, n), index=month_ends(n))
    fit = af.fit_decay(y, n_boot=200, rng=9)
    assert fit.better_fit == "linear"
    assert fit.aic_linear < fit.aic_exponential
    assert fit.linear_slope == pytest.approx(-0.002, rel=0.05)
    assert "linear model fits better" in fit.summary()


def test_rolling_input_gets_note_and_wide_blocks() -> None:
    rng = np.random.default_rng(1)
    n, k = 360, 200
    idx = month_ends(n)
    sig = pd.DataFrame(rng.standard_normal((n, k)), index=idx)
    fwd = 0.1 * sig + pd.DataFrame(rng.standard_normal((n, k)), index=idx)
    ric = af.rolling_ic(sig, fwd, window=36)
    fit = af.fit_decay(ric, n_boot=200, rng=1)
    assert fit.block_size >= 36
    assert any("rolling series" in note for note in fit.notes)
    assert "Note:" in fit.summary()


def test_reproducible_with_seed() -> None:
    ic = decaying_ic(40, 0.02, 0)
    a = af.fit_decay(ic, n_boot=200, rng=42)
    b = af.fit_decay(ic, n_boot=200, rng=42)
    assert a.rate_ci == b.rate_ci
    c = af.fit_decay(ic, n_boot=200, rng=np.random.default_rng(42))
    assert c.rate_ci == a.rate_ci


def test_interior_nans_warn_and_are_dropped() -> None:
    ic = decaying_ic(40, 0.01, 0)
    ic.iloc[[10, 20]] = np.nan
    with pytest.warns(af.DataDroppedWarning, match="Dropped 2"):
        fit = af.fit_decay(ic, n_boot=100, rng=0)
    assert fit.n_obs == 478


def test_short_sample_note_and_explicit_block() -> None:
    ic = decaying_ic(3, 0.01, 0)  # 36 obs
    fit = af.fit_decay(ic, n_boot=100, block_size=12, rng=0)
    assert fit.block_size == 12
    assert any("independent blocks" in note for note in fit.notes)


def test_summary_mentions_half_life_and_ci() -> None:
    fit = af.fit_decay(decaying_ic(60, 0.005, 3), n_boot=200, rng=3)
    text = fit.summary()
    assert "half-life" in text
    assert "95% CI" in text
    assert "per year" in text


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_boot": 50}, "n_boot"),
        ({"n_boot": 2.5}, "n_boot"),
        ({"alpha": 0.7}, "alpha"),
        ({"block_size": 0}, "block_size"),
        ({"block_size": 1.5}, "block_size"),
        ({"rng": "x"}, "rng"),
    ],
)
def test_argument_validation(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(af.InputError, match=match):
        af.fit_decay(decaying_ic(10, 0.01, 0), **kwargs)  # type: ignore[arg-type]


def test_too_short_or_constant_raises() -> None:
    with pytest.raises(af.InsufficientDataError, match="at least 20"):
        af.fit_decay(decaying_ic(1, 0.01, 0))
    with pytest.raises(af.InputError, match="constant"):
        af.fit_decay(pd.Series(0.03, index=month_ends(60)))


def test_linear_fallback_ci_accounts_for_starting_level_uncertainty() -> None:
    # Regression test for the review finding: bootstrap rates must vary the intercept too.
    from alphafade.decay import _boot_linear

    t = np.arange(120) / 12
    fitted = 1.0 - 0.05 * t
    resid = np.random.default_rng(0).normal(0, 0.3, 120)
    b0, b1 = _boot_linear(t, fitted, resid, 5, 400, np.random.default_rng(1))
    assert b0.std() > 0.01
    # Each (b0*, b1*) pair is the OLS fit of its own bootstrap sample.
    assert np.corrcoef(b0, b1)[0, 1] < -0.5  # intercept and slope co-vary, as in OLS
