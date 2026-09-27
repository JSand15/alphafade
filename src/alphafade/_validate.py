"""Input checks, alignment, and frequency inference.

Everything user-facing funnels through these helpers so the rules are the same everywhere:
inputs must have a sorted, unique DatetimeIndex, nothing is reindexed or resampled behind
the user's back, and anything lossy emits a :class:`DataDroppedWarning` with counts.
"""

from __future__ import annotations

import warnings
from typing import Final, Literal

import numpy as np
import pandas as pd

from ._errors import (
    AlignmentError,
    DataDroppedWarning,
    FrequencyError,
    InputError,
    InsufficientDataError,
)

Freq = Literal["D", "W", "M", "Q", "A"]

PERIODS_PER_YEAR: Final[dict[str, int]] = {"D": 252, "W": 52, "M": 12, "Q": 4, "A": 1}

# Accepted spellings for ``freq=`` arguments.
_FREQ_ALIASES: Final[dict[str, Freq]] = {
    "d": "D",
    "b": "D",
    "daily": "D",
    "w": "W",
    "weekly": "W",
    "m": "M",
    "me": "M",
    "ms": "M",
    "monthly": "M",
    "q": "Q",
    "qe": "Q",
    "quarterly": "Q",
    "a": "A",
    "y": "A",
    "ye": "A",
    "annual": "A",
    "yearly": "A",
}

# Allowed gap (in calendar days) between consecutive observations for each frequency.
# Bands don't overlap. Longer holiday or closure gaps (e.g. the 6-day 2001 NYSE closure) fall
# outside the band but cover a tiny share of the time span, so they're tolerated below.
_GAP_BANDS: Final[dict[Freq, tuple[float, float]]] = {
    "D": (1.0, 5.0),
    "W": (5.5, 10.0),
    "M": (25.0, 35.0),
    "Q": (80.0, 100.0),
    "A": (350.0, 380.0),
}

# Share of the total time span that may be covered by out-of-band gaps before the index is
# called irregular or mixed-frequency.
_MAX_IRREGULAR_SHARE: Final = 0.05

DAYS_PER_YEAR: Final = 365.25


def _check_index(index: pd.Index, name: str) -> pd.DatetimeIndex:
    if isinstance(index, pd.MultiIndex):
        raise InputError(
            f"{name} has a MultiIndex. alphafade expects wide data: one row per date and one "
            f"column per asset. Try `{name}.unstack()` to reshape it."
        )
    if not isinstance(index, pd.DatetimeIndex):
        raise InputError(
            f"{name} must have a DatetimeIndex, got {type(index).__name__}. "
            f"Convert it with `{name}.index = pd.to_datetime({name}.index)`."
        )
    if index.tz is not None:
        raise InputError(
            f"{name} has a timezone-aware index ({index.tz}). Remove the timezone with "
            f"`{name}.index = {name}.index.tz_localize(None)`."
        )
    if index.has_duplicates:
        dupes = index[index.duplicated()].unique()[:3]
        raise InputError(
            f"{name} has duplicate dates (e.g. {[str(d.date()) for d in dupes]}). "
            "Each date must appear once; aggregate or drop the duplicates first."
        )
    if not index.is_monotonic_increasing:
        raise InputError(f"{name} dates are not sorted. Sort them with `{name}.sort_index()`.")
    return index


def as_series(x: object, name: str) -> pd.Series[float]:
    """Coerce ``x`` to a float Series with a validated DatetimeIndex."""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise InputError(
                f"{name} must be a Series (one return per date), got a DataFrame with "
                f"{x.shape[1]} columns. Pick one column, e.g. `{name}['my_strategy']`."
            )
        x = x.iloc[:, 0]
    if not isinstance(x, pd.Series):
        raise InputError(f"{name} must be a pandas Series, got {type(x).__name__}.")
    _check_index(x.index, name)
    try:
        out = x.astype("float64")
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must contain numbers: {exc}") from None
    if np.isinf(out.to_numpy()).any():
        raise InputError(f"{name} contains infinite values. Replace or remove them first.")
    return out


def as_panel(x: object, name: str) -> pd.DataFrame:
    """Coerce ``x`` to a float DataFrame (dates x assets) with a validated index."""
    if isinstance(x, pd.Series):
        raise InputError(
            f"{name} must be a DataFrame with one column per asset, got a Series. "
            "If your data is in long format, reshape it with `.unstack()`."
        )
    if not isinstance(x, pd.DataFrame):
        raise InputError(f"{name} must be a pandas DataFrame, got {type(x).__name__}.")
    _check_index(x.index, name)
    if x.columns.has_duplicates:
        raise InputError(f"{name} has duplicate column (asset) labels.")
    try:
        out = x.astype("float64")
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must contain only numbers: {exc}") from None
    if np.isinf(out.to_numpy()).any():
        raise InputError(f"{name} contains infinite values. Replace or remove them first.")
    return out


def as_timestamp(date: object, name: str) -> pd.Timestamp:
    """Parse a user-supplied date."""
    try:
        ts = pd.Timestamp(date)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise InputError(f"{name}={date!r} is not a valid date. Use e.g. '1993-03-01'.") from None
    if pd.isna(ts):
        raise InputError(f"{name} must be a date, got NaT.")
    return ts


def check_window(window: int, n: int, name: str = "window") -> int:
    """Validate a rolling-window length against the number of observations."""
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
        raise InputError(f"{name} must be an integer number of periods, got {window!r}.")
    if window < 2:
        raise InputError(f"{name} must be at least 2, got {window}.")
    if window > n:
        raise InsufficientDataError(
            f"{name}={window} is longer than the data ({n} observations). "
            "Use a shorter window or more data."
        )
    return int(window)


def parse_freq(freq: str) -> Freq:
    """Normalize a user-supplied frequency string."""
    key = str(freq).strip().lower()
    if key not in _FREQ_ALIASES:
        raise FrequencyError(
            f"Unknown freq {freq!r}. Use one of 'D' (daily), 'W' (weekly), 'M' (monthly), "
            "'Q' (quarterly), or 'A' (annual)."
        )
    return _FREQ_ALIASES[key]


def infer_freq(index: pd.DatetimeIndex, name: str) -> Freq:
    """Infer D/W/M/Q/A from the spacing of dates, refusing to guess on mixed data.

    The frequency whose allowed gap band contains the median gap wins. If gaps outside that
    band cover more than 5% of the total time span, the data is called mixed or irregular
    and a :class:`FrequencyError` is raised instead of guessing.
    """
    if len(index) < 3:
        raise FrequencyError(
            f"Can't infer the frequency of {name} from {len(index)} dates. Pass freq= "
            "explicitly ('D', 'W', 'M', 'Q' or 'A')."
        )
    gaps = np.diff(_as_ns(index)) / (86_400 * 1e9)
    median = float(np.median(gaps))
    match: Freq | None = None
    for freq, (lo, hi) in _GAP_BANDS.items():
        if lo <= median <= hi:
            match = freq
            break
    if match is None:
        raise FrequencyError(
            f"Can't infer the frequency of {name}: the typical gap between dates is "
            f"{median:.1f} days, which isn't daily, weekly, monthly, quarterly or annual. "
            "Resample the data or pass freq= explicitly."
        )
    lo, hi = _GAP_BANDS[match]
    off_band = (gaps < lo) | (gaps > hi)
    share = float(gaps[off_band].sum() / gaps.sum())
    if share > _MAX_IRREGULAR_SHARE:
        raise FrequencyError(
            f"{name} looks like mixed or irregular frequency: it is mostly {match!r} but "
            f"{share:.0%} of its time span has gaps that don't fit. Resample it to one "
            "frequency (e.g. `.resample('ME').sum()` for monthly) or pass freq= explicitly."
        )
    return match


def resolve_freq(index: pd.DatetimeIndex, freq: str | None, name: str) -> Freq:
    """Return ``freq`` if the user gave one, otherwise infer it."""
    if freq is not None:
        return parse_freq(freq)
    return infer_freq(index, name)


def same_freq(a: pd.DatetimeIndex, b: pd.DatetimeIndex, name_a: str, name_b: str) -> Freq:
    """Infer both frequencies and raise if they differ."""
    fa = infer_freq(a, name_a)
    fb = infer_freq(b, name_b)
    if fa != fb:
        raise FrequencyError(
            f"{name_a} is {fa!r} but {name_b} is {fb!r}. alphafade never resamples for you: "
            f"convert one of them first so both have the same frequency."
        )
    return fa


def dropna_series(x: pd.Series[float], name: str) -> pd.Series[float]:
    """Drop NaNs, warning about interior gaps.

    Leading and trailing NaNs (e.g. from rolling warm-up or forward-return shifts) carry no
    information, so they're trimmed silently. NaNs in the middle are real missing data, so
    dropping them emits a :class:`DataDroppedWarning`.
    """
    valid = x.notna().to_numpy()
    if not valid.any():
        raise InsufficientDataError(f"{name} has no non-missing values.")
    first = int(np.argmax(valid))
    last = len(valid) - int(np.argmax(valid[::-1]))
    trimmed = x.iloc[first:last]
    interior = int(trimmed.isna().sum())
    if interior:
        warnings.warn(
            f"Dropped {interior} missing value(s) from the middle of {name} "
            f"({interior / len(trimmed):.1%} of observations).",
            DataDroppedWarning,
            stacklevel=3,
        )
    return trimmed.dropna()


def align_panels(
    a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict two panels to their shared dates and assets, warning about what's left out."""
    dates = a.index.intersection(b.index)
    assets = a.columns.intersection(b.columns)
    if len(dates) == 0:
        raise AlignmentError(
            f"{name_a} and {name_b} share no dates. Check that both use the same date "
            f"convention (e.g. both month-end). {name_a} runs {_span(a.index)}, {name_b} "
            f"runs {_span(b.index)}."
        )
    if len(assets) == 0:
        raise AlignmentError(
            f"{name_a} and {name_b} share no asset (column) labels. Check that both use "
            "the same identifiers (tickers, PERMNOs, ...)."
        )
    dropped_dates = len(a.index.union(b.index)) - len(dates)
    dropped_assets = len(a.columns.union(b.columns)) - len(assets)
    if dropped_dates or dropped_assets:
        warnings.warn(
            f"{name_a} and {name_b} only partly overlap: using {len(dates)} shared dates and "
            f"{len(assets)} shared assets; ignoring {dropped_dates} date(s) and "
            f"{dropped_assets} asset(s) that appear in only one of them.",
            DataDroppedWarning,
            stacklevel=3,
        )
    return a.loc[dates, assets], b.loc[dates, assets]


def years_since_start(index: pd.DatetimeIndex) -> np.ndarray:
    """Elapsed calendar time in years from the first date (uses real dates, not row counts)."""
    ns = _as_ns(index).astype("float64")
    out: np.ndarray = (ns - ns[0]) / (86_400 * 1e9 * DAYS_PER_YEAR)
    return out


def _as_ns(index: pd.DatetimeIndex) -> np.ndarray:
    # pandas 3 may store datetimes in s/ms/us resolution; normalize to integer nanoseconds.
    return index.as_unit("ns").to_numpy().astype(np.int64)


def _span(index: pd.Index) -> str:
    if len(index) == 0:
        return "(empty)"
    return f"{index[0]:%Y-%m-%d} to {index[-1]:%Y-%m-%d}"
