from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import alphafade as af
from alphafade.crowding import _leave_one_out, _pairwise

WEEKS = pd.date_range("1990-01-05", periods=520, freq="W-FRI")


def legs(
    tickers: list[int], dates: pd.DatetimeIndex, split: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    is_long = np.array([i < split for i in range(len(tickers))])
    long_ = pd.DataFrame(np.tile(is_long, (len(dates), 1)), index=dates, columns=tickers)
    return long_, ~long_


def test_independent_stocks_score_near_zero(rng: np.random.Generator) -> None:
    rets = pd.DataFrame(rng.normal(0, 0.03, (520, 60)), index=WEEKS)
    long_, short_ = legs(list(range(60)), WEEKS[51::13], 30)
    crowd = af.crowding_score(rets, long_, short_)
    assert crowd["long"].notna().all()
    assert abs(crowd["long"].mean()) < 0.03
    assert abs(crowd["short"].mean()) < 0.03
    assert (crowd["n_long"] == 30).all()


def test_rising_common_loading_makes_score_trend_up(rng: np.random.Generator) -> None:
    n, k = 520, 40
    common = rng.normal(0, 0.02, n)
    loading = np.linspace(0.0, 1.5, n)  # capital piling in over time
    rets = pd.DataFrame(rng.normal(0, 0.03, (n, k)) + (loading * common)[:, None], index=WEEKS)
    long_, short_ = legs(list(range(k)), WEEKS[51::13], 20)
    crowd = af.crowding_score(rets, long_, short_)
    trend = (
        crowd["long"]
        .reset_index(drop=True)
        .corr(pd.Series(np.arange(len(crowd)), dtype=float), method="spearman")
    )
    assert trend > 0.9
    assert crowd["long"].iloc[-1] > crowd["long"].iloc[0] + 0.3
    assert crowd["mean"].iloc[-1] == pytest.approx(crowd[["long", "short"]].iloc[-1].mean())


def test_factor_adjustment_removes_explained_comovement(rng: np.random.Generator) -> None:
    n, k = 520, 40
    mkt = rng.normal(0.002, 0.02, n)
    rf = np.full(n, 0.0005)
    betas = rng.uniform(0.5, 1.5, k)
    rets = pd.DataFrame(
        rf[:, None] + np.outer(mkt, betas) + rng.normal(0, 0.02, (n, k)), index=WEEKS
    )
    factors = pd.DataFrame({"Mkt-RF": mkt, "RF": rf}, index=WEEKS)
    long_, short_ = legs(list(range(k)), WEEKS[51::26], 20)
    raw = af.crowding_score(rets, long_, short_)
    adjusted = af.crowding_score(rets, long_, short_, factors=factors)
    assert raw["long"].mean() > 0.5
    assert abs(adjusted["long"].mean()) < 0.05


def test_leave_one_out_and_pairwise_match_manual(rng: np.random.Generator) -> None:
    resid = rng.normal(size=(52, 4)) + rng.normal(size=(52, 1))
    manual_loo = np.mean(
        [
            np.corrcoef(resid[:, i], np.delete(resid, i, axis=1).mean(axis=1))[0, 1]
            for i in range(4)
        ]
    )
    c = np.corrcoef(resid, rowvar=False)
    manual_pair = c[np.triu_indices(4, 1)].mean()
    assert _leave_one_out(resid) == pytest.approx(manual_loo, rel=1e-12)
    assert _pairwise(resid) == pytest.approx(manual_pair, rel=1e-12)
    # With missing values, pairwise uses pairwise-complete observations like pandas.
    holed = resid.copy()
    holed[:5, 0] = np.nan
    expected = pd.DataFrame(holed).corr().to_numpy()[np.triu_indices(4, 1)].mean()
    assert _pairwise(holed) == pytest.approx(expected, rel=1e-12)
    loo_holed = _leave_one_out(holed)
    assert np.isfinite(loo_holed)


def test_end_to_end_matches_manual_residual_computation(rng: np.random.Generator) -> None:
    n, k = 60, 6
    idx = WEEKS[:n]
    fac = pd.DataFrame({"Mkt-RF": rng.normal(0, 0.02, n)}, index=idx)
    rets = pd.DataFrame(rng.normal(0, 0.03, (n, k)) + fac.to_numpy() * 0.8, index=idx)
    long_, short_ = legs(list(range(k)), idx[[-1]], 3)
    for method in ("leave_one_out", "pairwise"):
        crowd = af.crowding_score(
            rets, long_, short_, factors=fac, window=52, method=method, min_stocks=3
        )
        win = rets.iloc[-52:, :3].to_numpy()
        x = np.column_stack([np.ones(52), fac.iloc[-52:].to_numpy()])
        resid = win - x @ np.linalg.lstsq(x, win, rcond=None)[0]
        expected = _leave_one_out(resid) if method == "leave_one_out" else _pairwise(resid)
        assert crowd["long"].iloc[0] == pytest.approx(expected, rel=1e-10)


def test_no_look_ahead_uses_window_ending_at_formation(rng: np.random.Generator) -> None:
    rets = pd.DataFrame(rng.normal(0, 0.03, (120, 10)), index=WEEKS[:120])
    long_, short_ = legs(list(range(10)), WEEKS[[60]], 5)
    base = af.crowding_score(rets, long_, short_)
    changed = rets.copy()
    changed.iloc[61:] = rng.normal(0, 0.03, (59, 10))  # alter only the future
    assert af.crowding_score(changed, long_, short_)["long"].iloc[0] == base["long"].iloc[0]


def test_warmup_dates_are_nan_without_warning(rng: np.random.Generator) -> None:
    rets = pd.DataFrame(rng.normal(0, 0.03, (100, 10)), index=WEEKS[:100])
    long_, short_ = legs(list(range(10)), WEEKS[[10, 70]], 5)
    crowd = af.crowding_score(rets, long_, short_)
    assert np.isnan(crowd["long"].iloc[0])
    assert np.isfinite(crowd["long"].iloc[1])
    assert crowd["n_long"].iloc[0] == 0


def test_thin_stocks_and_legs_warn(rng: np.random.Generator) -> None:
    rets = pd.DataFrame(rng.normal(0, 0.03, (100, 10)), index=WEEKS[:100])
    rets.iloc[:90, 0] = np.nan  # stock 0 has only 10 obs
    long_, short_ = legs(list(range(10)), WEEKS[[80]], 5)
    with pytest.warns(af.DataDroppedWarning, match="Skipped 1 stock-window"):
        crowd = af.crowding_score(rets, long_, short_, min_stocks=3)
    assert crowd["n_long"].iloc[0] == 4
    with pytest.warns(af.DataDroppedWarning) as rec:
        crowd = af.crowding_score(rets, long_, short_, min_stocks=5)
    assert any("fewer than 5 usable stocks" in str(w.message) for w in rec)
    assert np.isnan(crowd["long"].iloc[0])


def test_partial_data_stock_still_used(rng: np.random.Generator) -> None:
    rets = pd.DataFrame(rng.normal(0, 0.03, (100, 10)), index=WEEKS[:100])
    rets.iloc[60:70, 0] = np.nan  # 42 of 52 obs present, above min_obs=26
    long_, short_ = legs(list(range(10)), WEEKS[[99]], 5)
    crowd = af.crowding_score(rets, long_, short_)
    assert crowd["n_long"].iloc[0] == 5
    assert np.isfinite(crowd["long"].iloc[0])


def test_unknown_members_warn_or_raise(rng: np.random.Generator) -> None:
    rets = pd.DataFrame(rng.normal(0, 0.03, (100, 10)), index=WEEKS[:100])
    long_, short_ = legs(list(range(12)), WEEKS[[80]], 5)  # short leg: 7 names, 2 unknown
    with pytest.warns(af.DataDroppedWarning, match="2 member stock"):
        af.crowding_score(rets, long_, short_)
    bad_long, bad_short = legs([100, 101, 102], WEEKS[[80]], 2)
    with pytest.raises(af.AlignmentError, match="same identifiers"):
        af.crowding_score(rets, bad_long, bad_short)


def test_factor_validation(rng: np.random.Generator) -> None:
    rets = pd.DataFrame(rng.normal(0, 0.03, (100, 10)), index=WEEKS[:100])
    long_, short_ = legs(list(range(10)), WEEKS[[80]], 5)
    with pytest.raises(af.AlignmentError, match="missing"):
        af.crowding_score(
            rets, long_, short_, factors=pd.DataFrame({"m": 0.0}, index=WEEKS[5:100])
        )
    monthly = pd.DataFrame({"m": 0.0}, index=pd.date_range("1990-01-31", periods=40, freq="ME"))
    with pytest.raises(af.FrequencyError):
        af.crowding_score(rets, long_, short_, factors=monthly)
    holed = pd.DataFrame({"m": rng.normal(size=100)}, index=WEEKS[:100])
    holed.iloc[3] = np.nan
    with pytest.raises(af.InputError, match="missing values"):
        af.crowding_score(rets, long_, short_, factors=holed)


def test_argument_validation(rng: np.random.Generator) -> None:
    rets = pd.DataFrame(rng.normal(0, 0.03, (100, 10)), index=WEEKS[:100])
    long_, short_ = legs(list(range(10)), WEEKS[[80]], 5)
    with pytest.raises(af.InputError, match="method"):
        af.crowding_score(rets, long_, short_, method="kendall")  # type: ignore[arg-type]
    with pytest.raises(af.InputError, match="min_obs"):
        af.crowding_score(rets, long_, short_, min_obs=2)
    with pytest.raises(af.InputError, match="exceed window"):
        af.crowding_score(rets, long_, short_, window=20, min_obs=26)
    with pytest.raises(af.InputError, match="boolean DataFrame"):
        af.crowding_score(rets, [1, 2], short_)  # type: ignore[arg-type]
    with pytest.raises(af.InputError, match="True/False"):
        af.crowding_score(rets, long_.astype(float) * 3, short_)
    with pytest.raises(af.AlignmentError, match="same formation dates"):
        af.crowding_score(rets, long_, legs(list(range(10)), WEEKS[[81]], 5)[1])
    nan_members = long_.astype(object)
    nan_members.iloc[0, 0] = np.nan
    # NaN membership counts as False.
    crowd = af.crowding_score(rets, nan_members.astype(float), short_, min_stocks=3)
    assert crowd["n_long"].iloc[0] == 4
