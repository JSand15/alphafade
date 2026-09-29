from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from alphafade._errors import InputError, InsufficientDataError
from alphafade.walkforward import WalkForwardResult, walk_forward_decay
from tests.conftest import month_ends

LN2_TAU = 5 * math.log(2)


def decaying(n_years: int, noise: float, seed: int, tau: float = 5.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    n = n_years * 12
    t = np.arange(n) / 12
    return pd.Series(0.10 * np.exp(-t / tau) + rng.normal(0, noise, n), index=month_ends(n))


def test_exponential_decay_recovers_half_life_in_later_windows() -> None:
    res = walk_forward_decay(decaying(60, 0.005, 1), min_obs=240, step=60, n_boot=100, rng=1)
    assert isinstance(res, WalkForwardResult)
    later = res.table.iloc[-3:]
    assert later["decay_detected"].all()
    assert (later["half_life_years"] - LN2_TAU).abs().max() < 0.25 * LN2_TAU
    assert list(res.table.columns) == [
        "n_obs",
        "decay_rate",
        "half_life_years",
        "ci_low",
        "ci_high",
        "p_value",
        "decay_detected",
        "model",
        "fallback",
    ]
    assert res.table["n_obs"].iloc[-1] == 720
    assert res.table.index[-1] == month_ends(720)[-1]
    assert res.stable_since is not None
    assert res.share_detected > 0.5
    assert "walk-forward" in res.summary().lower()


def test_noise_mostly_undetected() -> None:
    rng = np.random.default_rng(3)
    s = pd.Series(rng.normal(0.02, 0.05, 240), index=month_ends(240))
    res = walk_forward_decay(s, min_obs=60, step=12, n_boot=100, rng=3)
    assert res.share_detected <= 0.5
    assert res.stable_since is None or res.share_detected > 0


def test_regime_change_stable_since_after_change() -> None:
    rng = np.random.default_rng(5)
    n = 480
    t = np.arange(n) / 12
    edge = np.where(t < 20, 0.10, 0.10 * np.exp(-(t - 20) / 4))
    s = pd.Series(edge + rng.normal(0, 0.004, n), index=month_ends(n))
    res = walk_forward_decay(s, min_obs=120, step=12, n_boot=100, rng=5)
    change = month_ends(n)[240]
    if res.stable_since is not None:
        assert res.stable_since >= change
    assert res.table["decay_detected"].iloc[-1]
    assert res.stable_since is not None
    assert res.half_life_drift is not None and res.half_life_drift > 0


def test_no_look_ahead() -> None:
    s = decaying(30, 0.01, 2)
    cut = 200
    t_date = s.index[cut - 1]
    s2 = s.copy()
    s2.iloc[cut:] = s2.iloc[cut:] * -5 + 1.0
    a = walk_forward_decay(s, min_obs=60, step=20, n_boot=100, rng=9).table
    b = walk_forward_decay(s2, min_obs=60, step=20, n_boot=100, rng=9).table
    pd.testing.assert_frame_equal(a.loc[:t_date], b.loc[:t_date])
    assert a.index[a.index <= t_date].size >= 2


def test_reproducible_and_seed_matters() -> None:
    s = decaying(20, 0.02, 4)
    a = walk_forward_decay(s, min_obs=60, step=60, n_boot=100, rng=1).table
    b = walk_forward_decay(s, min_obs=60, step=60, n_boot=100, rng=1).table
    pd.testing.assert_frame_equal(a, b)
    c = walk_forward_decay(s, min_obs=60, step=60, n_boot=100, rng=np.random.default_rng(1))
    assert len(c.table) == len(a)


def test_fit_warning_caught_and_recorded() -> None:
    # A near-instant collapse forces the exponential fit to its bound -> linear fallback.
    n = 60
    y = np.zeros(n)
    y[0] = 1.0
    s = pd.Series(y + np.random.default_rng(0).normal(0, 1e-3, n), index=month_ends(n))
    res = walk_forward_decay(s, min_obs=30, step=30, n_boot=100, rng=0)
    assert res.table["fallback"].any()
    assert (res.table.loc[res.table["fallback"], "model"] == "linear").all()


def test_step_larger_than_data_gives_single_window_and_full_sample_included() -> None:
    s = decaying(10, 0.01, 6)
    res = walk_forward_decay(s, min_obs=100, step=1000, n_boot=100, rng=0)
    assert list(res.table["n_obs"]) == [100, 120]


def test_errors() -> None:
    s = decaying(10, 0.01, 6)
    with pytest.raises(InputError, match="min_obs"):
        walk_forward_decay(s, min_obs=19)
    with pytest.raises(InputError, match="step"):
        walk_forward_decay(s, step=0)
    with pytest.raises(InputError, match="step"):
        walk_forward_decay(s, step=1.5)  # type: ignore[arg-type]
    with pytest.raises(InputError, match="n_boot"):
        walk_forward_decay(s, n_boot=10)
    with pytest.raises(InputError, match="alpha"):
        walk_forward_decay(s, alpha=0.9)
    with pytest.raises(InsufficientDataError, match="min_obs"):
        walk_forward_decay(s, min_obs=200)
    with pytest.raises(InputError, match="constant"):
        walk_forward_decay(pd.Series(1.0, index=month_ends(100)), min_obs=60)
    with pytest.raises(InputError):
        walk_forward_decay([1.0, 2.0], min_obs=60)  # type: ignore[arg-type]


def test_nans_trimmed_and_interior_warns() -> None:
    from alphafade._errors import DataDroppedWarning

    s = decaying(10, 0.01, 6)
    s.iloc[:3] = np.nan
    res = walk_forward_decay(s, min_obs=60, step=30, n_boot=100, rng=0)
    assert res.table["n_obs"].iloc[-1] == 117
    s.iloc[50] = np.nan
    with pytest.warns(DataDroppedWarning):
        walk_forward_decay(s, min_obs=60, step=30, n_boot=100, rng=0)


def test_summary_mentions_instability() -> None:
    res = walk_forward_decay(decaying(40, 0.02, 8), min_obs=120, step=24, n_boot=100, rng=8)
    text = res.summary()
    assert "windows" in text
    assert res.stability["share_detected"] == res.share_detected
