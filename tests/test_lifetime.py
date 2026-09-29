from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from alphafade import DecayFit, InputError, fit_decay
from alphafade.lifetime import LifetimeResult, signal_lifetime
from tests.conftest import month_ends

LAM = math.log(2) / 5  # half-life of exactly 5 years


def make_fit(
    *,
    model: str = "exponential",
    rate: float = LAM,
    rate_ci: tuple[float, float] | None = None,
    initial: float = 0.10,
    detected: bool = True,
    start: str = "1980-01-31",
    end: str = "2000-01-31",
) -> DecayFit:
    """A hand-built DecayFit so expected answers can be computed on paper."""
    rate_ci = rate_ci if rate_ci is not None else (rate * 0.8, rate * 1.25)
    half = (math.log(2) if model == "exponential" else 0.5) / rate if detected else None
    idx = pd.date_range(start, end, freq="ME")
    return DecayFit(
        model=model,  # type: ignore[arg-type]
        decay_detected=detected,
        half_life_years=half,
        ci_low=None,
        ci_high=None,
        decay_rate=rate,
        rate_ci=rate_ci,
        p_value=0.0,
        initial_level=initial,
        r2=0.5,
        r2_exponential=0.5,
        r2_linear=0.4,
        aic_exponential=0.0,
        aic_linear=1.0,
        better_fit="exponential",
        linear_slope=-rate * initial,
        n_obs=len(idx),
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
        block_size=3,
        alpha=0.05,
        fitted=pd.Series(np.zeros(len(idx)), index=idx),
    )


def test_hand_computed_exponential_floor() -> None:
    res = signal_lifetime(make_fit(), floor=0.025)
    assert isinstance(res, LifetimeResult)
    assert res.years_from_start == pytest.approx(10.0)  # ln(0.10/0.025)/lambda = 2 half-lives
    assert res.reason is None
    assert res.floor == pytest.approx(0.025)


def test_from_end_remaining_and_date() -> None:
    res = signal_lifetime(make_fit(), floor=0.025)  # sample runs ~20 years
    span = (pd.Timestamp("2000-01-31") - pd.Timestamp("1980-01-31")).days / 365.25
    assert res.years_remaining == 0.0  # 10 years < 20-year span: already there
    late = signal_lifetime(make_fit(), floor=0.01)  # ln(10)/lambda = 16.6 years, still < span
    assert late.years_remaining == 0.0
    far = signal_lifetime(make_fit(), floor=0.001)  # ln(100)/lambda = 33.2 years
    assert far.years_from_start == pytest.approx(math.log(100) / LAM)
    assert far.years_remaining == pytest.approx(far.years_from_start - span)
    assert far.date is not None
    expected = pd.Timestamp("1980-01-31") + pd.Timedelta(days=far.years_from_start * 365.25)
    assert abs((far.date - expected).days) <= 1


def test_from_end_false_skips_remaining() -> None:
    res = signal_lifetime(make_fit(), floor=0.001, from_end=False)
    assert res.years_remaining is None
    assert res.years_from_start is not None


def test_fraction_half_reproduces_half_life_exactly() -> None:
    fit = make_fit()
    res = signal_lifetime(fit, fraction=0.5)
    assert res.years_from_start == fit.half_life_years
    lin = make_fit(model="linear", rate=0.04)
    assert signal_lifetime(lin, fraction=0.5).years_from_start == lin.half_life_years


def test_fraction_and_floor_agree() -> None:
    a = signal_lifetime(make_fit(), fraction=0.25)
    b = signal_lifetime(make_fit(), floor=0.025)
    assert a.years_from_start == pytest.approx(b.years_from_start)
    assert a.fraction == 0.25


def test_ci_ordering_and_values() -> None:
    res = signal_lifetime(make_fit(), floor=0.025)
    assert res.ci_low is not None and res.ci_high is not None
    assert res.ci_low < res.years_from_start < res.ci_high  # type: ignore[operator]
    # Faster decay (upper rate bound) gives the earlier date.
    assert res.ci_low == pytest.approx(10.0 / 1.25)
    assert res.ci_high == pytest.approx(10.0 / 0.8)


def test_linear_case_hand_computed() -> None:
    # edge(t) = 0.10 - 0.002 t  => rate = 0.02 of the starting level per year.
    fit = make_fit(model="linear", rate=0.02, rate_ci=(0.015, 0.03))
    res = signal_lifetime(fit, floor=0.025)
    assert res.years_from_start == pytest.approx((0.10 - 0.025) / 0.002)  # 37.5
    assert res.ci_low == pytest.approx(0.75 / 0.03)
    assert res.ci_high == pytest.approx(0.75 / 0.015)
    # A linear line does cross zero, unlike an exponential.
    zero = signal_lifetime(fit, floor=0.0)
    assert zero.years_from_start == pytest.approx(50.0)
    assert signal_lifetime(fit, fraction=0.25).years_from_start == pytest.approx(0.75 / 0.02)


def test_no_decay_detected_returns_none_with_reason() -> None:
    fit = make_fit(detected=False, rate=0.01, rate_ci=(-0.02, 0.04))
    res = signal_lifetime(fit, floor=0.02)
    assert res.years_from_start is None
    assert res.years_remaining is None and res.date is None
    assert res.ci_low is None and res.ci_high is None
    assert res.reason is not None and "no detectable decay" in res.reason.lower()
    assert "No answer" in res.summary()


def test_floor_at_or_above_start_is_already_below() -> None:
    for floor in (0.10, 0.5):
        res = signal_lifetime(make_fit(), floor=floor)
        assert res.years_from_start == 0.0
        assert res.years_remaining == 0.0
        assert res.date == pd.Timestamp("1980-01-31")
        assert res.reason is not None and "already" in res.reason
        assert (res.ci_low, res.ci_high) == (0.0, 0.0)
    lin = signal_lifetime(make_fit(model="linear", rate=0.02), floor=0.2)
    assert lin.years_from_start == 0.0


@pytest.mark.parametrize("floor", [0.0, -0.01])
def test_exponential_never_reaches_nonpositive_floor(floor: float) -> None:
    res = signal_lifetime(make_fit(), floor=floor)
    assert res.years_from_start is None
    assert res.reason is not None and "never" in res.reason


def test_nonpositive_starting_level_is_not_answerable() -> None:
    res = signal_lifetime(make_fit(initial=-0.05), floor=0.01)
    assert res.years_from_start is None
    assert res.reason is not None and "not positive" in res.reason


def test_absurdly_far_date_is_none_not_error() -> None:
    fit = make_fit(rate=1e-9, rate_ci=(5e-10, 2e-9))
    res = signal_lifetime(fit, floor=0.05)
    assert res.years_from_start is not None and res.years_from_start > 1e8
    assert res.date is None
    assert "beyond" in res.summary()


def test_summary_readable() -> None:
    text = signal_lifetime(make_fit(), floor=0.001).summary()
    assert "33.2" in text and "years" in text and "95% CI" in text
    assert "starting level" in text  # the held-fixed simplification is disclosed


def test_result_is_frozen() -> None:
    res = signal_lifetime(make_fit(), floor=0.025)
    with pytest.raises(AttributeError):
        res.floor = 1.0  # type: ignore[misc]
    assert replace(res, floor=1.0).floor == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"floor": 0.02, "fraction": 0.5},
        {"fraction": 0.0},
        {"fraction": 1.0},
        {"fraction": 1.5},
        {"fraction": -0.2},
        {"floor": float("nan")},
        {"floor": float("inf")},
        {"floor": True},
        {"fraction": True},
        {"floor": "0.02"},
    ],
)
def test_bad_arguments_raise_input_error(kwargs: dict[str, object]) -> None:
    with pytest.raises(InputError):
        signal_lifetime(make_fit(), **kwargs)  # type: ignore[arg-type]


def test_non_decayfit_raises() -> None:
    with pytest.raises(InputError, match="DecayFit"):
        signal_lifetime(pd.Series([1.0, 2.0]), floor=0.01)  # type: ignore[arg-type]


def test_end_to_end_recovers_true_lifetime() -> None:
    tau, a, floor = 5.0, 0.10, 0.025
    n = 80 * 12
    t = np.arange(n) / 12
    rng = np.random.default_rng(7)
    ic = pd.Series(a * np.exp(-t / tau) + rng.normal(0, 0.005, n), index=month_ends(n))
    fit = fit_decay(ic, n_boot=300, rng=7)
    res = signal_lifetime(fit, floor=floor)
    truth = tau * math.log(a / floor)  # 6.93 years
    assert res.years_from_start == pytest.approx(truth, rel=0.10)
    assert res.ci_low is not None and res.ci_high is not None
    assert res.ci_low < truth < res.ci_high
    half = signal_lifetime(fit, fraction=0.5)
    assert half.years_from_start == fit.half_life_years
    assert half.years_remaining == 0.0
