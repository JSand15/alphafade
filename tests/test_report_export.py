from __future__ import annotations

import copy
import json
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

import alphafade as af
from alphafade.report import _jsonable

IDX = pd.date_range("1965-01-31", "2014-12-31", freq="ME")
T = np.arange(len(IDX)) / 12


def strategy(seed: int = 0, tau: float = 12.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(0.012 * np.exp(-T / tau) + rng.normal(0, 0.01, len(IDX)), index=IDX)


def panel(seed: int = 0, k: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rho = 0.10 * np.exp(-T / 8)
    sig = rng.standard_normal((len(IDX), k))
    fwd = rho[:, None] * sig + np.sqrt(1 - rho[:, None] ** 2) * rng.standard_normal((len(IDX), k))
    return pd.DataFrame(sig, index=IDX), pd.DataFrame(fwd, index=IDX)


@pytest.fixture(scope="module")
def full() -> af.FadeReport:
    sig, fwd = panel()
    crowd = pd.Series(T / T[-1] + np.random.default_rng(1).normal(0, 0.05, len(IDX)), index=IDX)
    return af.analyze(
        strategy(),
        signal=sig,
        fwd_returns=fwd,
        sample_end="1989-12-31",
        publication_date="1993-03-01",
        crowding=crowd,
        n_boot=100,
        rng=0,
    )


@pytest.fixture(scope="module")
def minimal() -> af.FadeReport:
    r = pd.Series(np.random.default_rng(3).normal(0.005, 0.02, len(IDX)), index=IDX)
    return af.analyze(r, publication_date="1993-03-01", n_boot=100, rng=3)


def _strict_loads(text: str) -> Any:
    def bad(token: str) -> None:
        raise AssertionError(f"non-strict JSON token {token}")

    return json.loads(text, parse_constant=bad)


@pytest.mark.parametrize("fixture", ["full", "minimal"])
@pytest.mark.parametrize("series", [False, True])
def test_json_round_trip_is_strict(
    fixture: str, series: bool, request: pytest.FixtureRequest
) -> None:
    rep = request.getfixturevalue(fixture)
    text = rep.to_json(include_series=series)
    assert "NaN" not in text and "Infinity" not in text
    assert _strict_loads(text) == json.loads(json.dumps(rep.to_dict(include_series=series)))
    compact = rep.to_json(indent=None)
    assert "\n" not in compact
    assert _strict_loads(compact) == _strict_loads(rep.to_json())


def test_values_match_attributes(full: af.FadeReport) -> None:
    d = full.to_dict()
    assert d["verdict"] == full.verdict()
    assert (d["freq"], d["window"], d["n_obs"]) == ("M", 36, len(full.returns))
    assert d["start"] == "1965-01-31"
    assert d["end"] == "2014-12-31"
    assert d["sample_end"] == "1989-12-31"
    assert d["publication_date"] == "1993-03-01"
    assert full.ic is not None and full.ic_decay is not None
    assert d["mean_ic"] == pytest.approx(float(full.ic.mean()))
    rd = d["return_decay"]
    fit = full.return_decay
    assert rd["half_life_years"] == fit.half_life_years
    assert rd["ci_low"] == fit.ci_low
    assert rd["decay_rate"] == fit.decay_rate
    assert rd["rate_ci"] == list(fit.rate_ci)
    assert rd["model"] == fit.model
    assert rd["better_fit"] == fit.better_fit
    assert rd["decay_detected"] is bool(fit.decay_detected)
    assert rd["notes"] == list(fit.notes)
    assert rd["start"] == fit.start.date().isoformat()
    assert "fitted" not in rd
    assert d["ic_decay"]["n_obs"] == full.ic_decay.n_obs
    brk = d["break_test"]
    assert brk["date"] == full.break_test.date.date().isoformat()
    assert brk["stat"] == full.break_test.stat
    assert brk["method"] == full.break_test.method
    assert "path" not in brk
    assert full.publication is not None and full.crowding_link is not None
    gap = d["publication"]
    assert gap["post_publication_decline"] == pytest.approx(
        full.publication.post_publication_decline
    )
    assert isinstance(gap["table"], list)
    assert len(gap["table"]) == len(full.publication.table)
    assert d["publication_test"] is None
    link = d["crowding_link"]
    assert link["horizon"] == 12
    assert link["t_stat"] == full.crowding_link.t_stat
    assert link["n_obs"] == full.crowding_link.n_obs
    assert "series" not in d


def test_optional_parts_absent(minimal: af.FadeReport) -> None:
    d = minimal.to_dict()
    assert minimal.publication_test is not None
    assert d["ic_decay"] is None
    assert d["mean_ic"] is None
    assert d["publication"] is None
    assert d["crowding_link"] is None
    assert d["sample_end"] is None
    assert d["publication_test"]["method"] == "chow"
    assert d["publication_test"]["date"] == minimal.publication_test.date.date().isoformat()


def test_nan_and_inf_become_null() -> None:
    assert _jsonable(float("nan")) is None
    assert _jsonable(np.float64("inf")) is None
    assert _jsonable(np.int64(3)) == 3
    assert type(_jsonable(np.int64(3))) is int
    assert _jsonable((1.0, math.nan)) == [1.0, None]
    assert _jsonable(pd.NaT) is None
    assert _jsonable(np.bool_(True)) is True
    assert _jsonable(pd.Timestamp("2020-05-31")) == "2020-05-31"


def test_include_series_adds_keys(full: af.FadeReport) -> None:
    base = full.to_dict()
    d = full.to_dict(include_series=True)
    assert "series" in d and "series" not in base
    assert set(d["series"]) == {"returns", "rolling_sharpe", "ic", "rolling_ic", "crowding"}
    assert len(d["series"]["returns"]) == len(full.returns)
    assert d["series"]["returns"]["1965-01-31"] == float(full.returns.iloc[0])
    assert "fitted" in d["return_decay"]
    assert len(d["return_decay"]["fitted"]) == full.return_decay.n_obs
    assert "path" in d["break_test"]
    for k in ("verdict", "freq", "n_obs"):
        assert d[k] == base[k]


def test_series_only_for_present_parts(minimal: af.FadeReport) -> None:
    d = minimal.to_dict(include_series=True)
    assert set(d["series"]) == {"returns", "rolling_sharpe"}


def test_to_frame_headline_metrics(full: af.FadeReport, minimal: af.FadeReport) -> None:
    df = full.to_frame()
    assert set(df.columns) >= {"section", "metric", "value"}
    metrics = set(zip(df["section"], df["metric"], strict=True))
    for key in (
        ("return_decay", "half_life_years"),
        ("ic_decay", "decay_rate"),
        ("break_test", "p_value"),
        ("publication", "post_publication_t"),
        ("crowding_link", "t_stat"),
    ):
        assert key in metrics
    assert "crowding_link" not in set(minimal.to_frame()["section"])


def test_exports_do_not_mutate(full: af.FadeReport) -> None:
    assert full.publication is not None
    before_dict = copy.deepcopy(full.to_dict(include_series=True))
    r_before = full.returns.copy()
    table_before = full.publication.table.copy()
    fit_before = full.return_decay.fitted.copy()
    summary_before = full.summary()
    frame_before = full.to_frame()
    full.to_dict()
    full.to_json(include_series=True)
    pd.testing.assert_series_equal(full.returns, r_before)
    pd.testing.assert_frame_equal(full.publication.table, table_before)
    pd.testing.assert_series_equal(full.return_decay.fitted, fit_before)
    assert full.summary() == summary_before
    pd.testing.assert_frame_equal(full.to_frame(), frame_before)
    assert full.to_dict(include_series=True) == before_dict
