from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import alphafade as af

END = "1989-12-31"
PUB = "1993-03-01"


def mp_series(sd: float = 0.002, seed: int = 0, start: str = "1965-01-31") -> pd.Series:
    """Returns with McLean & Pontiff's pattern: 26% lower post-sample, 58% lower post-pub."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, "2012-12-31", freq="ME")
    mean = np.where(idx <= END, 0.010, np.where(idx < PUB, 0.0074, 0.0042))
    return pd.Series(mean + rng.normal(0, sd, len(idx)), index=idx)


def test_recovers_mclean_pontiff_declines() -> None:
    gap = af.publication_gap(mp_series(), sample_end=END, publication_date=PUB)
    assert gap.post_sample_decline == pytest.approx(0.26, abs=0.03)
    assert gap.post_publication_decline == pytest.approx(0.58, abs=0.02)
    assert gap.in_sample_mean == pytest.approx(0.010, abs=5e-4)
    assert gap.post_publication_t < -10
    assert gap.publication_vs_post_sample_t < -3
    assert gap.freq == "M"
    t = gap.table
    assert list(t.index) == ["in_sample", "post_sample", "post_publication"]
    assert t.loc["in_sample", "end"] == pd.Timestamp("1989-12-31")
    assert t.loc["post_sample", "start"] == pd.Timestamp("1990-01-31")
    assert t.loc["post_publication", "start"] == pd.Timestamp("1993-03-31")
    assert t.loc["in_sample", "n_obs"] == 300
    assert t.loc["post_sample", "decline"] == pytest.approx(gap.post_sample_decline)
    assert t.loc["in_sample", "ann_mean"] == pytest.approx(t.loc["in_sample", "mean"] * 12)


def test_regression_matches_statsmodels() -> None:
    r = mp_series(sd=0.03, seed=1)
    gap = af.publication_gap(r, sample_end=END, publication_date=PUB, hac_lags=5)
    idx = r.index
    x = np.column_stack(
        [
            np.ones(len(r)),
            ((idx > END) & (idx < PUB)).astype(float),
            (idx >= PUB).astype(float),
        ]
    )
    ref = sm.OLS(r.to_numpy(), x).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    assert gap.in_sample_mean == pytest.approx(ref.params[0], rel=1e-10)
    assert gap.post_sample_change == pytest.approx(ref.params[1], rel=1e-10)
    assert gap.post_sample_t == pytest.approx(ref.tvalues[1], rel=1e-10)
    assert gap.post_publication_t == pytest.approx(ref.tvalues[2], rel=1e-10)
    diff = ref.t_test(np.array([[0.0, -1.0, 1.0]]))
    assert gap.publication_vs_post_sample_t == pytest.approx(diff.tvalue.item(), rel=1e-8)


def test_summary_is_plain_english() -> None:
    text = af.publication_gap(mp_series(), sample_end=END, publication_date=PUB).summary()
    assert "In-sample" in text
    assert re.search(r"5[6-9]% lower than in-sample", text)
    assert "publication itself seems to matter" in text


def test_sample_start_drops_with_warning() -> None:
    r = mp_series(start="1960-01-31")
    with pytest.warns(af.DataDroppedWarning, match="Dropped 60"):
        gap = af.publication_gap(
            r, sample_end=END, publication_date=PUB, sample_start="1965-01-01"
        )
    assert gap.table.loc["in_sample", "n_obs"] == 300
    with pytest.raises(af.InputError, match="before sample_end"):
        af.publication_gap(r, sample_end=END, publication_date=PUB, sample_start="1995-01-01")


def test_no_post_sample_period() -> None:
    r = mp_series()
    gap = af.publication_gap(r, sample_end="1989-12-31", publication_date="1990-01-01")
    assert gap.table.loc["post_sample", "n_obs"] == 0
    assert math.isnan(gap.post_sample_decline)
    assert math.isnan(gap.publication_vs_post_sample_t)
    assert "no observations" in gap.summary()


def test_nonpositive_in_sample_mean_gives_nan_declines() -> None:
    r = -mp_series()
    gap = af.publication_gap(r, sample_end=END, publication_date=PUB)
    assert math.isnan(gap.post_publication_decline)
    assert "not meaningful" in gap.summary()


def test_validation() -> None:
    r = mp_series()
    with pytest.raises(af.InputError, match="must be after"):
        af.publication_gap(r, sample_end=PUB, publication_date=END)
    with pytest.raises(af.InsufficientDataError, match="in-sample"):
        af.publication_gap(r, sample_end="1965-03-31", publication_date=PUB)
    with pytest.raises(af.InsufficientDataError, match="publication_date"):
        af.publication_gap(r, sample_end=END, publication_date="2020-01-01")


def test_daily_frequency_annualizes_with_252() -> None:
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("1985-01-01", "2000-12-31")
    r = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
    gap = af.publication_gap(r, sample_end="1990-12-31", publication_date="1994-01-01")
    assert gap.freq == "D"
    row = gap.table.loc["post_publication"]
    assert row["ann_mean"] == pytest.approx(row["mean"] * 252)
