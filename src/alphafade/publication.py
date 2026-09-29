"""McLean & Pontiff-style publication-gap analysis."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._errors import DataDroppedWarning, InputError, InsufficientDataError
from ._stats import hac_mean, ols_hac, resolve_hac_lags
from ._validate import (
    PERIODS_PER_YEAR,
    as_series,
    as_timestamp,
    dropna_series,
    resolve_freq,
    to_array,
)
from .rolling import WINDOW_ATTR

__all__ = ["GapResult", "publication_gap"]

_MIN_T_OBS = 12  # fewest observations for which a per-period t-stat is reported
PERIODS = ("in_sample", "post_sample", "post_publication")
_LABELS = {
    "in_sample": "In-sample",
    "post_sample": "After the sample ended, before publication",
    "post_publication": "After publication",
}


@dataclass(frozen=True)
class GapResult:
    """Result of :func:`publication_gap`.

    Attributes
    ----------
    table : DataFrame
        One row per period (``in_sample``, ``post_sample``, ``post_publication``) with
        columns ``start``, ``end``, ``n_obs``, ``mean``, ``ann_mean``, ``ann_sharpe``,
        ``t_stat`` (Newey-West t of the period mean), and ``decline`` (share of the in-sample
        mean lost; 0.58 means 58% lower). Empty periods have NaN values.
    in_sample_mean : float
    post_sample_change, post_publication_change : float
        Regression coefficients: how much lower (negative) or higher the average is in each
        later period than in-sample, in per-period return units.
    post_sample_t, post_publication_t : float
        Newey-West t-statistics of those changes.
    publication_vs_post_sample_t : float
        t-statistic for "post-publication differs from post-sample", i.e. whether
        publication itself mattered beyond ordinary out-of-sample decay.
    post_sample_decline, post_publication_decline : float
        The changes as a share of the in-sample mean (McLean & Pontiff report about 0.26 and
        0.58). NaN when the in-sample mean isn't positive.
    freq : str
    hac_lags : int
    """

    table: pd.DataFrame
    in_sample_mean: float
    post_sample_change: float
    post_sample_t: float
    post_publication_change: float
    post_publication_t: float
    publication_vs_post_sample_t: float
    post_sample_decline: float
    post_publication_decline: float
    freq: str
    hac_lags: int

    def summary(self) -> str:
        """Plain-English description of the result."""
        lines = []
        for period in PERIODS:
            row = self.table.loc[period].to_dict()
            if row["n_obs"] == 0:
                lines.append(f"{_LABELS[period]}: no observations.")
                continue
            text = (
                f"{_LABELS[period]} ({row['start']:%Y-%m} to {row['end']:%Y-%m}, "
                f"{int(row['n_obs'])} obs): average {row['mean']:.4g} per period "
                f"({row['ann_mean']:.2%} a year, Sharpe {row['ann_sharpe']:.2f}, "
                f"t = {row['t_stat']:.2f})"
            )
            if period != "in_sample" and not math.isnan(row["decline"]):
                change_t = (
                    self.post_sample_t if period == "post_sample" else self.post_publication_t
                )
                word = "lower" if row["decline"] >= 0 else "higher"
                text += (
                    f", {abs(row['decline']):.0%} {word} than in-sample "
                    f"(t of the change = {change_t:.2f})"
                )
            lines.append(text + ".")
        if not math.isnan(self.publication_vs_post_sample_t):
            lines.append(
                "Publication vs. post-sample difference: t = "
                f"{self.publication_vs_post_sample_t:.2f}"
                + (
                    " (publication itself seems to matter)."
                    if abs(self.publication_vs_post_sample_t) >= 2
                    else " (not distinguishable from ordinary out-of-sample decay)."
                )
            )
        if not self.in_sample_mean > 0:
            lines.append(
                "Note: the in-sample average isn't positive, so percentage declines are "
                "not meaningful."
            )
        return "\n".join(lines)


def publication_gap(
    returns: pd.Series[float],
    sample_end: str | pd.Timestamp,
    publication_date: str | pd.Timestamp,
    *,
    sample_start: str | pd.Timestamp | None = None,
    freq: str | None = None,
    hac_lags: int | None = None,
) -> GapResult:
    """Compare a strategy's returns in-sample, after the sample ended, and after publication.

    Following McLean & Pontiff (2016), the return series is split into three periods:

    * in-sample: up to and including ``sample_end`` (the data the original paper used),
    * post-sample: after ``sample_end`` but before ``publication_date``,
    * post-publication: from ``publication_date`` on.

    It then runs the regression r_t = a + b1 * post_sample_t + b2 * post_publication_t with
    Newey-West standard errors. ``a`` is the in-sample mean, and ``-b1 / a`` and ``-b2 / a``
    are the percentage declines.

    Parameters
    ----------
    returns : Series
        Per-period long-short strategy returns (decimals), indexed by date.
    sample_end : str or Timestamp
        Last date of the original study's sample.
    publication_date : str or Timestamp
        Date the result became public (journal publication, or first working paper if you
        prefer). Must be after ``sample_end``.
    sample_start : str or Timestamp, optional
        First date of the original sample. Earlier observations are dropped with a warning.
    freq : str, optional
        'D', 'W', 'M', 'Q' or 'A' for annualizing. Inferred if omitted.
    hac_lags : int, optional
        Newey-West lags, used for the regression and each period's t-stat (capped at the
        period length minus 1). Default: floor(4 * (n/100)^(2/9)) for the sample in
        question, and at least window - 1 for a rolling series.

    Returns
    -------
    GapResult

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> idx = pd.date_range("1965-01-31", "2010-12-31", freq="ME")
    >>> mean = np.where(idx <= "1989-12-31", 0.010, np.where(idx < "1993-03-01", 0.0074, 0.0042))
    >>> r = pd.Series(mean + rng.normal(0, 0.0002, len(idx)), index=idx)
    >>> gap = publication_gap(r, sample_end="1989-12-31", publication_date="1993-03-01")
    >>> round(gap.post_sample_decline, 2), round(gap.post_publication_decline, 2)
    (0.26, 0.58)
    """
    series = as_series(returns, "returns")
    end = as_timestamp(sample_end, "sample_end")
    pub = as_timestamp(publication_date, "publication_date")
    if pub <= end:
        raise InputError(
            f"publication_date ({pub:%Y-%m-%d}) must be after sample_end ({end:%Y-%m-%d})."
        )
    window = series.attrs.get(WINDOW_ATTR)
    y_s = dropna_series(series, "returns")
    f = resolve_freq(pd.DatetimeIndex(y_s.index), freq, "returns")
    if sample_start is not None:
        start = as_timestamp(sample_start, "sample_start")
        if start >= end:
            raise InputError("sample_start must be before sample_end.")
        before = int((y_s.index < start).sum())
        if before:
            warnings.warn(
                f"Dropped {before} observation(s) before sample_start ({start:%Y-%m-%d}).",
                DataDroppedWarning,
                stacklevel=2,
            )
            y_s = y_s[y_s.index >= start]
    idx = pd.DatetimeIndex(y_s.index)
    period = np.where(idx <= end, 0, np.where(idx < pub, 1, 2))
    n = len(y_s)
    n_in = int((period == 0).sum())
    if n_in < 12:
        raise InsufficientDataError(
            f"Only {n_in} in-sample observations (on or before {end:%Y-%m-%d}); need at least "
            "12. Check sample_end and the start of your data."
        )
    if (period == 2).sum() == 0:
        raise InsufficientDataError(
            f"No observations on or after publication_date ({pub:%Y-%m-%d}); the data ends "
            f"{idx[-1]:%Y-%m-%d}."
        )
    y = to_array(y_s)
    # Overlapping (rolling) inputs need at least window - 1 lags, as in find_break.
    min_lags = int(window) - 1 if isinstance(window, (int, np.integer)) and window > 1 else 0
    lags = resolve_hac_lags(hac_lags, n, min_lags=min(min_lags, n - 1))
    ppy = PERIODS_PER_YEAR[f]

    present = [p for p in (1, 2) if (period == p).any()]
    x = np.column_stack([np.ones(n)] + [(period == p).astype(float) for p in present])
    res = ols_hac(y, x, lags)
    coef = {p: float(res.params[i + 1]) for i, p in enumerate(present)}
    tval = {p: float(res.tvalues[i + 1]) for i, p in enumerate(present)}
    a = float(res.params[0])
    if len(present) == 2:
        contrast = np.array([0.0, -1.0, 1.0])
        var = float(contrast @ res.cov @ contrast)
        diff_t = (coef[2] - coef[1]) / math.sqrt(var) if var > 0 else math.nan
    else:
        diff_t = math.nan

    def decline(change: float) -> float:
        return -change / a if a > 0 else math.nan

    rows = []
    for p in range(len(PERIODS)):
        yp = y[period == p]
        ip = idx[period == p]
        if len(yp) == 0:
            rows.append(
                {
                    "start": pd.NaT,
                    "end": pd.NaT,
                    "n_obs": 0,
                    "mean": math.nan,
                    "ann_mean": math.nan,
                    "ann_sharpe": math.nan,
                    "t_stat": math.nan,
                    "decline": math.nan,
                }
            )
            continue
        mean = float(yp.mean())
        sd = float(yp.std(ddof=1)) if len(yp) > 1 else math.nan
        # Below 12 observations the HAC t-stat is not trustworthy; report NaN, not a big number.
        if len(yp) >= _MIN_T_OBS:
            period_lags = (
                min(lags, len(yp) - 1)
                if hac_lags is not None
                else max(resolve_hac_lags(None, len(yp)), min(min_lags, len(yp) - 1))
            )
            t_stat = hac_mean(yp, period_lags)[2]
        else:
            t_stat = math.nan
        rows.append(
            {
                "start": ip[0],
                "end": ip[-1],
                "n_obs": len(yp),
                "mean": mean,
                "ann_mean": mean * ppy,
                "ann_sharpe": mean / sd * math.sqrt(ppy) if sd and sd > 0 else math.nan,
                "t_stat": t_stat,
                "decline": 0.0 if p == 0 else decline(coef[p]),
            }
        )
    table = pd.DataFrame(rows, index=pd.Index(PERIODS, name="period"))
    return GapResult(
        table=table,
        in_sample_mean=a,
        post_sample_change=coef.get(1, math.nan),
        post_sample_t=tval.get(1, math.nan),
        post_publication_change=coef[2],
        post_publication_t=tval[2],
        publication_vs_post_sample_t=diff_t,
        post_sample_decline=decline(coef[1]) if 1 in coef else math.nan,
        post_publication_decline=decline(coef[2]),
        freq=f,
        hac_lags=lags,
    )
