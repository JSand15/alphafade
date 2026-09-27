from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import alphafade as af
from alphafade import _validate as v
from tests.conftest import business_days, month_ends


class TestInferFreq:
    @pytest.mark.parametrize(
        ("index", "expected"),
        [
            (business_days(600), "D"),
            (pd.date_range("2000-01-01", periods=400, freq="D"), "D"),
            (pd.date_range("2000-01-07", periods=300, freq="W-FRI"), "W"),
            (month_ends(120), "M"),
            (pd.date_range("2000-01-01", periods=120, freq="MS"), "M"),
            (pd.date_range("2000-03-31", periods=40, freq="QE"), "Q"),
            (pd.date_range("1950-12-31", periods=40, freq="YE"), "A"),
        ],
    )
    def test_regular_frequencies(self, index: pd.DatetimeIndex, expected: str) -> None:
        assert v.infer_freq(index, "x") == expected

    def test_daily_with_holidays_and_a_closure_is_still_daily(self) -> None:
        idx = business_days(1000)
        # Knock out a week (like the 2001 market closure) and scattered holidays.
        keep = np.ones(len(idx), dtype=bool)
        keep[400:405] = False
        keep[::37] = False
        assert v.infer_freq(idx[keep], "x") == "D"

    def test_mixed_monthly_then_daily_raises(self) -> None:
        monthly = pd.date_range("1990-01-31", periods=60, freq="ME")
        daily = pd.bdate_range("1995-01-02", "2014-12-31")
        idx = monthly.append(daily)
        with pytest.raises(af.FrequencyError, match="mixed or irregular"):
            v.infer_freq(idx, "returns")

    def test_unrecognized_spacing_raises(self) -> None:
        idx = pd.date_range("2000-01-01", periods=50, freq="15D")
        with pytest.raises(af.FrequencyError, match="typical gap"):
            v.infer_freq(idx, "x")

    def test_too_few_dates(self) -> None:
        with pytest.raises(af.FrequencyError, match="Pass freq="):
            v.infer_freq(month_ends(2), "x")

    def test_explicit_freq_bypasses_inference(self) -> None:
        idx = pd.date_range("2000-01-01", periods=50, freq="15D")
        assert v.resolve_freq(idx, "monthly", "x") == "M"
        assert v.resolve_freq(idx, "ME", "x") == "M"
        with pytest.raises(af.FrequencyError, match="Unknown freq"):
            v.resolve_freq(idx, "fortnightly", "x")

    def test_same_freq_rejects_mismatch(self) -> None:
        with pytest.raises(af.FrequencyError, match="never resamples"):
            v.same_freq(month_ends(60), business_days(600), "a", "b")
        assert v.same_freq(month_ends(60), month_ends(30), "a", "b") == "M"

    def test_pandas3_non_ns_resolution(self) -> None:
        idx = month_ends(24).as_unit("s")
        assert v.infer_freq(idx, "x") == "M"
        yrs = v.years_since_start(idx)
        assert yrs[-1] == pytest.approx((idx[-1] - idx[0]).days / 365.25)


class TestCoercion:
    def test_series_requires_datetime_index(self) -> None:
        with pytest.raises(af.InputError, match="DatetimeIndex"):
            v.as_series(pd.Series([1.0, 2.0]), "returns")

    def test_series_rejects_unsorted_duplicates_tz_inf(self) -> None:
        idx = month_ends(3)
        with pytest.raises(af.InputError, match="not sorted"):
            v.as_series(pd.Series([1.0, 2, 3], index=idx[::-1]), "r")
        with pytest.raises(af.InputError, match="duplicate dates"):
            v.as_series(pd.Series([1.0, 2, 3], index=idx[[0, 0, 1]]), "r")
        with pytest.raises(af.InputError, match="timezone"):
            v.as_series(pd.Series([1.0, 2, 3], index=idx.tz_localize("UTC")), "r")
        with pytest.raises(af.InputError, match="infinite"):
            v.as_series(pd.Series([1.0, np.inf, 3], index=idx), "r")
        with pytest.raises(af.InputError, match="numbers"):
            v.as_series(pd.Series(["a", "b", "c"], index=idx), "r")
        with pytest.raises(af.InputError, match="pandas Series"):
            v.as_series([1, 2, 3], "r")

    def test_single_column_frame_becomes_series(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0]}, index=month_ends(2))
        assert isinstance(v.as_series(df, "r"), pd.Series)
        with pytest.raises(af.InputError, match="Pick one column"):
            v.as_series(pd.DataFrame({"x": [1.0], "y": [2.0]}, index=month_ends(1)), "r")

    def test_panel_checks(self) -> None:
        idx = month_ends(2)
        with pytest.raises(af.InputError, match="got a Series"):
            v.as_panel(pd.Series([1.0, 2.0], index=idx), "signal")
        long = pd.DataFrame(
            {"v": [1.0, 2.0]},
            index=pd.MultiIndex.from_tuples([(idx[0], "a"), (idx[0], "b")]),
        )
        with pytest.raises(af.InputError, match="unstack"):
            v.as_panel(long, "signal")
        with pytest.raises(af.InputError, match="duplicate column"):
            v.as_panel(pd.DataFrame([[1.0, 2.0]], index=idx[:1], columns=["a", "a"]), "s")
        with pytest.raises(af.InputError, match="infinite"):
            v.as_panel(pd.DataFrame([[1.0, np.inf]], index=idx[:1]), "s")
        with pytest.raises(af.InputError, match="only numbers"):
            v.as_panel(pd.DataFrame([["x", "y"]], index=idx[:1]), "s")
        with pytest.raises(af.InputError, match="pandas DataFrame"):
            v.as_panel([[1.0]], "s")

    def test_timestamp_and_window(self) -> None:
        assert v.as_timestamp("1993-03-01", "d") == pd.Timestamp("1993-03-01")
        with pytest.raises(af.InputError, match="not a valid date"):
            v.as_timestamp("not a date", "d")
        with pytest.raises(af.InputError, match="must be a date"):
            v.as_timestamp(None, "d")
        assert v.check_window(5, 10) == 5
        with pytest.raises(af.InputError, match="at least 2"):
            v.check_window(1, 10)
        with pytest.raises(af.InputError, match="integer"):
            v.check_window(2.5, 10)  # type: ignore[arg-type]
        with pytest.raises(af.InsufficientDataError, match="longer than the data"):
            v.check_window(11, 10)


class TestDropnaAndAlign:
    def test_edge_nans_trimmed_silently_interior_warns(self) -> None:
        s = pd.Series([np.nan, 1.0, np.nan, 2.0, np.nan], index=month_ends(5))
        with pytest.warns(af.DataDroppedWarning, match="Dropped 1 missing"):
            out = v.dropna_series(s, "ic")
        assert out.tolist() == [1.0, 2.0]

    def test_edge_only_nans_no_warning(self) -> None:
        s = pd.Series([np.nan, 1.0, 2.0, np.nan], index=month_ends(4))
        assert v.dropna_series(s, "ic").tolist() == [1.0, 2.0]

    def test_all_nan_raises(self) -> None:
        with pytest.raises(af.InsufficientDataError):
            v.dropna_series(pd.Series([np.nan, np.nan], index=month_ends(2)), "ic")

    def test_align_partial_overlap_warns(self) -> None:
        a = pd.DataFrame(1.0, index=month_ends(4), columns=["x", "y", "z"])
        b = pd.DataFrame(2.0, index=month_ends(5)[1:], columns=["y", "z", "w"])
        with pytest.warns(af.DataDroppedWarning, match="3 shared dates and 2 shared assets"):
            a2, b2 = v.align_panels(a, b, "signal", "fwd_returns")
        assert list(a2.columns) == ["y", "z"] and len(a2) == 3
        assert b2.index.equals(a2.index)

    def test_align_no_overlap_raises(self) -> None:
        a = pd.DataFrame(1.0, index=month_ends(3), columns=["x"])
        with pytest.raises(af.AlignmentError, match="share no dates"):
            v.align_panels(a, a.set_axis(month_ends(3, "2000-01-31")), "a", "b")
        with pytest.raises(af.AlignmentError, match="share no asset"):
            v.align_panels(a, a.set_axis(["q"], axis=1), "a", "b")
