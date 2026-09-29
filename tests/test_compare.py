from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from alphafade import DataDroppedWarning, FitWarning, InputError, InsufficientDataError
from alphafade import compare as _compare
from alphafade.compare import SignalComparison, _adjust_pvalues
from tests.conftest import month_ends

N = 480


def compare_signals(*args: object, **kwargs: object) -> SignalComparison:
    # Pure-noise columns sometimes trip the linear fallback (a FitWarning); that is expected.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FitWarning)
        return _compare.compare_signals(*args, **kwargs)  # type: ignore[arg-type]


def make_panel(seed: int = 3, taus: tuple[float, float] = (3.0, 8.0)) -> pd.DataFrame:
    g = np.random.default_rng(seed)
    t = np.arange(N) / 12
    cols = {
        "fast": 0.10 * np.exp(-t / taus[0]) + g.normal(0, 0.02, N),
        "noise1": g.normal(0, 0.02, N),
        "slow": 0.10 * np.exp(-t / taus[1]) + g.normal(0, 0.02, N),
        "noise2": g.normal(0, 0.02, N),
        "noise3": g.normal(0, 0.02, N),
    }
    return pd.DataFrame(cols, index=month_ends(N))


def test_holm_known_answer() -> None:
    out = _adjust_pvalues(np.array([0.01, 0.04, 0.03, 0.005]), "holm")
    np.testing.assert_allclose(out, [0.03, 0.06, 0.06, 0.02])


def test_bh_known_answer() -> None:
    out = _adjust_pvalues(np.array([0.01, 0.04, 0.03, 0.005]), "bh")
    np.testing.assert_allclose(out, [0.02, 0.04, 0.04, 0.02])


def test_none_is_identity_and_caps_and_monotone() -> None:
    p = np.array([0.5, 0.9, 0.2, 0.0, 1.0])
    np.testing.assert_allclose(_adjust_pvalues(p, "none"), p)
    for method in ("holm", "bh"):
        adj = _adjust_pvalues(p, method)  # type: ignore[arg-type]
        assert adj.max() <= 1.0
        assert (adj >= p - 1e-15).all()
        o = np.argsort(p)
        assert (np.diff(adj[o]) >= -1e-15).all()
    assert _adjust_pvalues(np.array([0.9, 0.8]), "holm").max() == 1.0


def test_two_decay_three_noise() -> None:
    res = compare_signals(make_panel(), n_boot=300, rng=1)
    t = res.table
    assert isinstance(res, SignalComparison)
    assert list(t.index[:2]) == ["fast", "slow"]
    assert t["decay_detected_adjusted"].tolist() == [True, True, False, False, False]
    assert t["rank"].tolist() == [1, 2, 3, 4, 5]
    assert t.loc["fast", "half_life_years"] < t.loc["slow", "half_life_years"]
    assert (t["p_adjusted"] >= t["p_value"]).all()
    assert set(res.fits) == set(t.index)
    assert res.adjust == "holm"
    # Undetected signals are ordered by raw p-value.
    tail = t.iloc[2:]["p_value"].to_numpy()
    assert (np.diff(tail) >= 0).all()
    assert t.iloc[2:]["half_life_years"].isna().all()
    text = res.summary()
    assert "2 still do" in text and "luck" in text and "Holm" in text


def test_deterministic_with_same_seed() -> None:
    a = compare_signals(make_panel(), n_boot=200, rng=5).table
    b = compare_signals(make_panel(), n_boot=200, rng=5).table
    pd.testing.assert_frame_equal(a, b)
    g1 = compare_signals(make_panel(), n_boot=200, rng=np.random.default_rng(9)).table
    g2 = compare_signals(make_panel(), n_boot=200, rng=np.random.default_rng(9)).table
    pd.testing.assert_frame_equal(g1, g2)


def test_ranking_independent_of_input_column_order() -> None:
    panel = make_panel()
    a = compare_signals(panel, n_boot=200, rng=2).table
    b = compare_signals(panel[list(panel.columns[::-1])], n_boot=200, rng=2).table
    assert list(a.index[:2]) == list(b.index[:2]) == ["fast", "slow"]
    assert set(a.index) == set(b.index)
    assert a.loc["fast", "half_life_years"] == pytest.approx(
        b.loc["fast", "half_life_years"], rel=0.1
    )


def test_bh_and_none_options() -> None:
    panel = make_panel()
    holm = compare_signals(panel, n_boot=200, rng=4, adjust="holm").table
    bh = compare_signals(panel, n_boot=200, rng=4, adjust="bh").table
    none = compare_signals(panel, n_boot=200, rng=4, adjust="none")
    assert (bh["p_adjusted"] <= holm["p_adjusted"] + 1e-12).all()
    np.testing.assert_allclose(none.table["p_adjusted"], none.table["p_value"])
    assert "no correction" in none.summary() and "Warning" in none.summary()


def test_skips_short_and_empty_columns_with_warning() -> None:
    panel = make_panel()
    panel["tiny"] = np.nan
    panel.iloc[-10:, panel.columns.get_loc("tiny")] = np.random.default_rng(0).normal(size=10)
    panel["empty"] = np.nan
    panel["late"] = panel["fast"]
    panel.iloc[:100, panel.columns.get_loc("late")] = np.nan  # leading NaNs trimmed, kept
    with pytest.warns(DataDroppedWarning, match="tiny, empty"):
        res = compare_signals(panel, n_boot=200, rng=1)
    assert res.skipped == ("tiny", "empty")
    assert "tiny" not in res.table.index and "late" in res.table.index
    assert res.table.loc["late", "n_obs"] == N - 100
    assert "Skipped" in res.summary()


def test_seed_of_a_column_unaffected_by_skipping_others() -> None:
    panel = make_panel()
    base = compare_signals(panel, n_boot=200, rng=8).fits["slow"]
    extra = panel.copy()
    extra["empty"] = np.nan
    with pytest.warns(DataDroppedWarning):
        with_skip = compare_signals(extra, n_boot=200, rng=8).fits["slow"]
    assert base.rate_ci == with_skip.rate_ci


def test_fewer_than_two_usable_columns_raises() -> None:
    panel = make_panel()[["fast", "noise1"]].copy()
    panel["noise1"] = np.nan
    with (
        pytest.warns(DataDroppedWarning),
        pytest.raises(InsufficientDataError, match="at least 2"),
    ):
        compare_signals(panel, n_boot=200)


def test_constant_column_skipped() -> None:
    panel = make_panel()
    panel["flat"] = 0.01
    with pytest.warns(DataDroppedWarning, match="flat"):
        res = compare_signals(panel, n_boot=200, rng=1)
    assert res.skipped == ("flat",)


@pytest.mark.parametrize(
    "kwargs",
    [{"adjust": "bonferroni"}, {"n_boot": 10}, {"alpha": 0.9}, {"rng": "x"}],
)
def test_bad_arguments(kwargs: dict[str, object]) -> None:
    with pytest.raises(InputError):
        compare_signals(make_panel(), **kwargs)  # type: ignore[arg-type]


def test_rejects_series_input() -> None:
    with pytest.raises(InputError, match="DataFrame"):
        compare_signals(make_panel()["fast"])  # type: ignore[arg-type]
