"""How a signal's predictive power fades across forecast horizon (not calendar time)."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ._errors import InputError, InsufficientDataError
from ._stats import FloatArray, hac_mean, resolve_hac_lags
from ._validate import as_panel, dropna_series, to_array
from .rolling import ICMethod, forward_returns, ic_series

__all__ = ["HorizonResult", "ic_by_horizon"]

_MIN_DATES = 10
_MIN_POSITIVE_POINTS = 3


@dataclass(frozen=True)
class HorizonResult:
    """Result of :func:`ic_by_horizon`.

    Attributes
    ----------
    table : DataFrame
        Indexed by ``horizon`` (periods ahead, ascending). Columns: ``mean_ic`` (average
        information coefficient, IC, across dates), ``t_stat`` (Newey-West t-statistic of that
        mean, a t-statistic that allows for overlapping windows), ``n_dates`` (dates with a
        usable IC) and ``ic_per_period`` (``mean_ic / horizon``, for a rough like-for-like
        comparison across horizons).
    half_life_periods : float or None
        The horizon at which the fitted mean IC has fallen to half of its fitted value at the
        shortest horizon in the fit. None (see ``reason``) unless the mean IC is positive at
        every horizon and significant (t >= 2) at the shortest one.
        This is a point estimate only: no confidence interval is claimed, because the fit uses
        a handful of strongly overlapping, correlated points and any interval would overstate
        what is known.
    reason : str or None
        Why ``half_life_periods`` is None; None when it is a number.
    peak_horizon : int
        Horizon with the largest (most positive) mean IC. Check ``table["t_stat"]``
        before reading anything into it; ``summary()`` says when no horizon is significant.
    """

    table: pd.DataFrame = field(repr=False)
    half_life_periods: float | None
    reason: str | None
    peak_horizon: int

    def summary(self) -> str:
        """Plain-English description of the result."""
        t = self.table
        lines = [
            f"Mean IC at {len(t)} forecast horizons ({int(t.index.min())} to "
            f"{int(t.index.max())} periods ahead):"
        ]
        for h, row in zip(t.index.tolist(), t.to_dict("records"), strict=True):
            lines.append(
                f"  {int(h):>3} ahead: IC {row['mean_ic']:+.3f} (t = {row['t_stat']:.1f}, "
                f"{int(row['n_dates'])} dates)"
            )
        significant = (t["t_stat"].abs() >= 2).any()
        if not significant:
            lines.append(
                "No horizon has a statistically significant IC (every |t| < 2), so the "
                "differences between horizons may be noise."
            )
        elif (t["mean_ic"] <= 0).all():
            strongest = int(t.index[int(np.argmax(t["mean_ic"].abs().to_numpy()))])
            lines.append(
                "Every mean IC is negative: the signal predicts returns in the OPPOSITE "
                f"direction, most strongly at a horizon of {strongest} period(s)."
            )
        else:
            lines.append(
                f"The signal predicts best at a horizon of {self.peak_horizon} period(s)."
            )
        if self.half_life_periods is not None:
            lines.append(
                f"Fitted half-life: the mean IC halves by about {self.half_life_periods:.1f} "
                "periods ahead. This is a point estimate with no confidence interval."
            )
        else:
            lines.append(f"No half-life reported: {self.reason}")
        return "\n".join(lines)


def _fit_half_life(horizons: FloatArray, mean_ic: FloatArray) -> tuple[float | None, str | None]:
    """Fit ``mean_ic = b * exp(-k * h)`` by log-linear least squares on the positive points."""
    pos = mean_ic > 0
    n_pos = int(pos.sum())
    if n_pos < _MIN_POSITIVE_POINTS:
        return None, (
            f"only {n_pos} horizon(s) have a positive mean IC; the exponential fit needs at "
            f"least {_MIN_POSITIVE_POINTS}."
        )
    h = horizons[pos]
    slope = float(np.polyfit(h, np.log(mean_ic[pos]), 1)[0])
    k = -slope
    if not k > 0:
        return None, "the mean IC is not decreasing with horizon (fitted rate is not positive)."
    return float(h.min()) + math.log(2.0) / k, None


def _check_horizons(horizons: object) -> list[int]:
    if isinstance(horizons, (str, bytes)) or not isinstance(horizons, (Sequence, np.ndarray)):
        raise InputError(
            f"horizons must be a sequence of positive integers such as (1, 3, 12), got "
            f"{horizons!r}."
        )
    out: list[int] = []
    for h in horizons:
        if isinstance(h, (bool, np.bool_)) or not isinstance(h, (int, np.integer)) or h < 1:
            raise InputError(f"horizons must all be positive integers, got {h!r}.")
        out.append(int(h))
    if not out:
        raise InputError("horizons must not be empty; try (1, 2, 3, 6, 12).")
    if len(set(out)) != len(out):
        raise InputError(f"horizons must be unique, got {sorted(out)}.")
    return sorted(out)


def ic_by_horizon(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    horizons: Sequence[int] = (1, 2, 3, 6, 12),
    *,
    method: ICMethod = "spearman",
    min_assets: int = 5,
    hac_lags: int | None = None,
) -> HorizonResult:
    """Measure how far ahead a signal still predicts (fading across forecast horizon).

    The rest of alphafade asks whether a signal's edge fades across *calendar time*. This
    asks a different question: for a fixed period, how does predictive power change with the
    *forecast horizon*, the number of periods ahead being predicted. For each horizon ``h``
    the signal at date t is compared with the compounded return over the next ``h`` periods,
    (t, t + h], using :func:`forward_returns` and :func:`ic_series`.

    Parameters
    ----------
    signal : DataFrame
        Dates x assets. Values known at each date.
    returns : DataFrame
        Dates x assets of realized simple per-period returns as decimals. Not already
        forward-shifted: the shift is done here.
    horizons : sequence of int, default (1, 2, 3, 6, 12)
        Non-empty, unique, positive. Sorted internally.
    method : {"spearman", "pearson"}, default "spearman"
    min_assets : int, default 5
        Dates with fewer usable assets get no IC.
    hac_lags : int, optional
        Newey-West lags for each t-statistic. Default: the usual plug-in rule, but at least
        ``h - 1``, because neighbouring ``h``-period windows share ``h - 1`` periods of
        returns and their ICs are therefore correlated. An explicit value below ``h - 1``
        raises :class:`InputError`, since it would overstate significance.

    Returns
    -------
    HorizonResult

    Raises
    ------
    InputError
        Bad ``horizons``, ``method``, ``min_assets`` or ``hac_lags``, or malformed panels.
    InsufficientDataError
        A horizon leaves fewer than 10 dates with a usable IC.

    Warns
    -----
    DataDroppedWarning
        As in :func:`ic_series`; identical warnings from different horizons are shown once.

    Notes
    -----
    Multi-period returns are cumulative, so even a signal that predicts only the very next
    period shows a mean IC that shrinks roughly like ``1 / sqrt(h)`` as ``h`` grows: the
    extra periods add noise the signal cannot explain. Read the trend in ``mean_ic`` together
    with ``ic_per_period`` and the t-statistics, and do not mistake this built-in shrinkage
    for a slow-fading signal. A persistent signal, in contrast, can show an IC that rises
    with ``h``.

    ``half_life_periods`` comes from fitting ``mean_ic(h) = b * exp(-k * h)`` by least squares
    on the log of the positive mean ICs (at least three needed). It is the horizon at which the
    fitted curve is half its value at the shortest fitted horizon, ``h_min + ln(2) / k``. It
    is None if the mean IC is not decreasing in ``h`` (``k <= 0``) or too few points are
    positive. No confidence interval is provided.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> g = np.random.default_rng(0)
    >>> dates = pd.date_range("2000-01-31", periods=120, freq="ME")
    >>> sig = pd.DataFrame(g.standard_normal((120, 30)), index=dates)
    >>> ret = 0.05 * (0.3 * sig.shift(1).fillna(0) + g.standard_normal((120, 30)))
    >>> res = ic_by_horizon(sig, ret, horizons=(1, 3, 6))
    >>> res.table.columns.tolist()
    ['mean_ic', 't_stat', 'n_dates', 'ic_per_period']
    >>> res.peak_horizon
    1
    """
    hs = _check_horizons(horizons)
    if method not in ("spearman", "pearson"):
        raise InputError(f"method must be 'spearman' or 'pearson', got {method!r}.")
    if isinstance(min_assets, bool) or not isinstance(min_assets, (int, np.integer)):
        raise InputError(f"min_assets must be an integer, got {min_assets!r}.")
    if min_assets < 2:
        raise InputError("min_assets must be at least 2 (a correlation needs 2 points).")
    sig = as_panel(signal, "signal")
    ret = as_panel(returns, "returns")
    # Build forward returns on the FULL return history and only then line them up with the
    # signal (ic_series does that). Aligning first would make "h periods ahead" count the
    # signal's dates, so a missing signal month or a quarterly signal would silently pair
    # each date with the wrong future return.

    rows: list[dict[str, float]] = []
    seen: set[tuple[type, str]] = set()
    for h in hs:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ic = ic_series(sig, forward_returns(ret, h), method=method, min_assets=min_assets)
            try:
                clean = dropna_series(ic, f"the IC at horizon {h}")
            except InsufficientDataError:
                clean = ic.dropna()
        for w in caught:
            key = (w.category, str(w.message))
            if key not in seen:
                seen.add(key)
                warnings.warn(w.message, w.category, stacklevel=2)
        n = len(clean)
        if n < _MIN_DATES:
            raise InsufficientDataError(
                f"At horizon {h} only {n} date(s) have a usable IC (need at least "
                f"{_MIN_DATES}). Use a shorter horizon or more history, or check that the "
                "panels have at least min_assets assets with a varying signal."
            )
        lags = resolve_hac_lags(hac_lags, n, min_lags=min(h - 1, n - 1))
        _, _, t_stat = hac_mean(to_array(clean), lags)
        mean_ic = float(ic.mean())  # same NaN-skipping mean as ic_series(...).mean()
        rows.append(
            {
                "mean_ic": mean_ic,
                "t_stat": t_stat,
                "n_dates": float(n),
                "ic_per_period": mean_ic / h,
            }
        )

    table = pd.DataFrame(rows, index=pd.Index(hs, name="horizon"))
    table["n_dates"] = table["n_dates"].astype("int64")
    mean_ic_arr = table["mean_ic"].to_numpy(dtype=np.float64)
    t_arr = table["t_stat"].to_numpy(dtype=np.float64)
    half_life: float | None
    reason: str | None
    if (mean_ic_arr <= 0).any():
        half_life, reason = (
            None,
            (
                "the mean IC is not positive at every horizon, so a single decay curve does "
                "not describe it."
            ),
        )
    elif not t_arr[0] >= 2:
        half_life, reason = (
            None,
            (
                f"the IC at the shortest horizon ({hs[0]}) is not statistically significant "
                "(t < 2), so there is no clear edge to measure the fade of."
            ),
        )
    else:
        half_life, reason = _fit_half_life(np.asarray(hs, dtype=np.float64), mean_ic_arr)
    return HorizonResult(
        table=table,
        half_life_periods=half_life,
        reason=reason,
        peak_horizon=int(hs[int(np.argmax(mean_ic_arr))]),
    )
