"""One call that runs every alphafade analysis and explains the result in plain English."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields, is_dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ._errors import InputError, InsufficientDataError
from ._stats import RngLike, ols_hac, resolve_hac_lags, resolve_rng
from ._validate import (
    PERIODS_PER_YEAR,
    as_series,
    as_timestamp,
    check_window,
    resolve_freq,
    to_array,
)
from .breaks import BreakResult, chow_test, find_break
from .decay import DecayFit, fit_decay
from .publication import GapResult, publication_gap
from .rolling import WINDOW_ATTR, ICMethod, forward_returns, ic_series, rolling_ic, rolling_sharpe

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["CrowdingLink", "FadeReport", "analyze"]

# Default rolling window: three years of periods.
_DEFAULT_WINDOW = {"D": 756, "W": 156, "M": 36, "Q": 12, "A": 5}


@dataclass(frozen=True)
class CrowdingLink:
    """Whether crowding has predicted the strategy's future returns.

    Attributes
    ----------
    horizon : int
        Periods ahead over which the future return is compounded.
    slope_per_sd : float
        Change in the future ``horizon``-period return for a one-standard-deviation rise in
        the crowding score. Negative means more crowding came before weaker returns.
    t_stat : float
        Newey-West t-statistic (lags at least ``horizon - 1`` because the future windows
        overlap).
    correlation : float
    n_obs : int
    """

    horizon: int
    slope_per_sd: float
    t_stat: float
    correlation: float
    n_obs: int

    def summary(self) -> str:
        """Plain-English description."""
        if abs(self.t_stat) < 2:
            verdict = "is not a statistically reliable predictor of"
        elif self.slope_per_sd < 0:
            verdict = "has predicted weaker"
        else:
            verdict = "has predicted stronger"
        return (
            f"Crowding {verdict} returns over the next {self.horizon} periods: a one-standard-"
            f"deviation rise in crowding goes with a {self.slope_per_sd:+.2%} change in the "
            f"future return (t = {self.t_stat:.2f}, {self.n_obs} observations)."
        )


@dataclass(frozen=True)
class FadeReport:
    """Everything :func:`analyze` computed. Each part is ``None`` if its inputs weren't given.

    Attributes
    ----------
    returns : Series
        The strategy returns analysed (after trimming leading/trailing NaNs).
    freq : str
    window : int
        Rolling window (periods) used for ``rolling_sharpe`` and ``rolling_ic``.
    rolling_sharpe : Series
    return_decay : DecayFit
        Decay of the average return itself (fitted on raw per-period returns).
    break_test : BreakResult
        Unknown-date sup-Wald search on the returns.
    ic : Series or None
        Per-date information coefficient (if ``signal`` was given).
    rolling_ic : Series or None
    ic_decay : DecayFit or None
        Decay of the per-date IC.
    publication : GapResult or None
        If both ``sample_end`` and ``publication_date`` were given.
    publication_test : BreakResult or None
        Chow-style test at ``publication_date`` (if only that date was given).
    crowding : Series or None
    crowding_link : CrowdingLink or None
    """

    returns: pd.Series[float] = field(repr=False)
    freq: str
    window: int
    rolling_sharpe: pd.Series[float] = field(repr=False)
    return_decay: DecayFit
    break_test: BreakResult
    ic: pd.Series[float] | None = field(default=None, repr=False)
    rolling_ic: pd.Series[float] | None = field(default=None, repr=False)
    ic_decay: DecayFit | None = None
    publication: GapResult | None = None
    publication_test: BreakResult | None = None
    publication_date: pd.Timestamp | None = None
    sample_end: pd.Timestamp | None = None
    crowding: pd.Series[float] | None = field(default=None, repr=False)
    crowding_link: CrowdingLink | None = None

    # ----------------------------------------------------------------------------------
    def verdict(self) -> str:
        """One-sentence answer to "is my signal dying?"."""
        main = self.ic_decay if self.ic_decay is not None else self.return_decay
        what = "signal's IC" if self.ic_decay is not None else "strategy's average return"
        if main.decay_detected and main.half_life_years is not None:
            text = (
                f"Yes: the {what} is fading, with a half-life of about "
                f"{main.half_life_years:.1f} years."
            )
        else:
            text = f"No detectable decay in the {what} so far."
        if self.break_test.significant():
            direction = "drop" if self.break_test.change < 0 else "jump"
            text += f" There was a significant {direction} around {self.break_test.date:%Y-%m}."
        if (
            self.publication is not None
            and not math.isnan(self.publication.post_publication_decline)
            and abs(self.publication.post_publication_t) >= 2
        ):
            text += (
                f" Returns are {self.publication.post_publication_decline:.0%} lower after "
                "publication."
            )
        if self.crowding_link is not None and abs(self.crowding_link.t_stat) >= 2:
            text += (
                " Crowding has predicted weaker returns."
                if self.crowding_link.slope_per_sd < 0
                else " Crowding has predicted stronger returns."
            )
        return text

    def _ic_window_note(self) -> str:
        """Say so when the IC window was capped below ``window`` by a short signal history."""
        if self.rolling_ic is None:
            return ""
        used = self.rolling_ic.attrs.get(WINDOW_ATTR)
        if used is None or used == self.window:
            return ""
        return f" (the IC rolling window is {used}, capped at the available IC dates)"

    def summary(self) -> str:
        """Plain-English report of every analysis that was run."""
        r = self.returns
        ppy = PERIODS_PER_YEAR[self.freq]
        sd = float(r.std(ddof=1))
        sharpe = float(r.mean()) / sd * math.sqrt(ppy) if sd > 0 else math.nan
        parts = [
            "alphafade report",
            "=" * 16,
            f"Verdict: {self.verdict()}",
            "",
            f"Data: {len(r)} {_FREQ_WORD[self.freq]} returns, {r.index[0]:%Y-%m-%d} to "
            f"{r.index[-1]:%Y-%m-%d}. Average {float(r.mean()) * ppy:.2%} a year, Sharpe "
            f"{sharpe:.2f}. Rolling window: {self.window} periods{self._ic_window_note()}.",
            "",
            "Decay of the average return",
            _indent(self.return_decay.summary()),
        ]
        if self.ic_decay is not None and self.ic is not None:
            parts += [
                "",
                f"Decay of the information coefficient (mean IC {float(self.ic.mean()):.4f})",
                _indent(self.ic_decay.summary()),
            ]
        parts += ["", "Structural break", _indent(self.break_test.summary())]
        if self.publication is not None:
            parts += ["", "Publication gap", _indent(self.publication.summary())]
        if self.publication_test is not None:
            parts += [
                "",
                "Change at the publication date",
                _indent(self.publication_test.summary()),
            ]
        if self.crowding_link is not None:
            parts += ["", "Crowding", _indent(self.crowding_link.summary())]
        return "\n".join(parts)

    def to_frame(self) -> pd.DataFrame:
        """All headline numbers as a tidy table with columns section, metric, value."""
        rows: list[tuple[str, str, Any]] = []

        def add_decay(section: str, fit: DecayFit) -> None:
            for name in (
                "model",
                "decay_detected",
                "half_life_years",
                "ci_low",
                "ci_high",
                "decay_rate",
                "p_value",
                "initial_level",
                "r2",
                "better_fit",
                "linear_slope",
                "n_obs",
            ):
                rows.append((section, name, getattr(fit, name)))

        add_decay("return_decay", self.return_decay)
        if self.ic_decay is not None:
            add_decay("ic_decay", self.ic_decay)
        for section, brk in (
            ("break_test", self.break_test),
            ("publication_test", self.publication_test),
        ):
            if brk is None:
                continue
            for name in ("date", "stat", "p_value", "mean_before", "mean_after", "change_t"):
                rows.append((section, name, getattr(brk, name)))
        if self.publication is not None:
            gap = self.publication
            for name in (
                "in_sample_mean",
                "post_sample_decline",
                "post_sample_t",
                "post_publication_decline",
                "post_publication_t",
                "publication_vs_post_sample_t",
            ):
                rows.append(("publication", name, getattr(gap, name)))
        if self.crowding_link is not None:
            for name in ("horizon", "slope_per_sd", "t_stat", "correlation", "n_obs"):
                rows.append(("crowding_link", name, getattr(self.crowding_link, name)))
        return pd.DataFrame(rows, columns=["section", "metric", "value"])

    def to_dict(self, *, include_series: bool = False) -> dict[str, Any]:
        """Headline results as a nested, JSON-safe dictionary.

        Only plain Python values appear: floats, ints, strings, bools, ``None``, lists and
        dicts. Dates become ISO strings (``"2015-12-31"``); NaN and infinity become ``None``
        (strict JSON has no way to write them); tuples become lists. Optional parts that were
        not computed are ``None``.

        Parameters
        ----------
        include_series : bool, default False
            Also include the big time series (returns, rolling Sharpe, IC, crowding, fitted
            curves, break-test path) as ``{iso_date: value}`` maps.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> idx = pd.date_range("2000-01-31", periods=120, freq="ME")
        >>> r = pd.Series(np.random.default_rng(0).normal(0.01, 0.02, 120), index=idx)
        >>> d = analyze(r, n_boot=100, rng=0).to_dict()
        >>> d["freq"], d["n_obs"], d["start"]
        ('M', 120, '2000-01-31')
        """
        r = self.returns
        out: dict[str, Any] = {
            "verdict": self.verdict(),
            "freq": self.freq,
            "window": self.window,
            "n_obs": len(r),
            "start": _jsonable(r.index[0]),
            "end": _jsonable(r.index[-1]),
            "publication_date": _jsonable(self.publication_date),
            "sample_end": _jsonable(self.sample_end),
            "mean_ic": _float_or_none(self.ic.mean()) if self.ic is not None else None,
            "return_decay": _obj_to_dict(self.return_decay, include_series),
            "ic_decay": _obj_to_dict(self.ic_decay, include_series),
            "break_test": _obj_to_dict(self.break_test, include_series),
            "publication": _obj_to_dict(self.publication, include_series),
            "publication_test": _obj_to_dict(self.publication_test, include_series),
            "crowding_link": _obj_to_dict(self.crowding_link, include_series),
        }
        if include_series:
            out["series"] = {
                name: _jsonable(s)
                for name, s in (
                    ("returns", self.returns),
                    ("rolling_sharpe", self.rolling_sharpe),
                    ("ic", self.ic),
                    ("rolling_ic", self.rolling_ic),
                    ("crowding", self.crowding),
                )
                if s is not None
            }
        return out

    def to_json(self, *, indent: int | None = 2, include_series: bool = False) -> str:
        """:meth:`to_dict` as a strict JSON string (no ``NaN`` or ``Infinity`` tokens)."""
        return json.dumps(
            self.to_dict(include_series=include_series), indent=indent, allow_nan=False
        )

    def plot(self, figsize: tuple[float, float] | None = None) -> Figure:
        """Draw every panel on one matplotlib Figure (needs ``pip install 'alphafade[plot]'``)."""
        from .plotting import plot_report

        return plot_report(self, figsize=figsize)


_FREQ_WORD = {"D": "daily", "W": "weekly", "M": "monthly", "Q": "quarterly", "A": "annual"}


def _float_or_none(x: Any) -> float | None:
    """Return a plain float, or None if the value is NaN or infinite."""
    v = float(x)
    return v if math.isfinite(v) else None


def _jsonable(v: Any) -> Any:
    """Convert numpy/pandas values to plain JSON-safe Python (NaN and inf become None)."""
    if v is None or v is pd.NaT:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return _float_or_none(v)
    if isinstance(v, str):
        return v
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        ts: Any = pd.Timestamp(v)
        return None if ts is pd.NaT else ts.date().isoformat()
    if isinstance(v, pd.Series):
        return {_key(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, pd.DataFrame):
        return [
            {str(c): _jsonable(x) for c, x in row.items()} for _, row in v.reset_index().iterrows()
        ]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_jsonable(x) for x in v]
    return str(v)


def _key(k: Any) -> str:
    return str(_jsonable(k)) if isinstance(k, (pd.Timestamp, np.datetime64)) else str(k)


def _obj_to_dict(obj: Any, include_series: bool) -> dict[str, Any] | None:
    """Turn a result dataclass into a dict using its fields, so new fields appear automatically.

    Pandas Series are skipped unless ``include_series``; small DataFrames (tables) are kept.
    """
    if obj is None:
        return None
    assert is_dataclass(obj)
    out: dict[str, Any] = {}
    for f in fields(obj):
        v = getattr(obj, f.name)
        if isinstance(v, pd.Series) and not include_series:
            continue
        out[f.name] = _jsonable(v)
    return out


def _indent(text: str) -> str:
    return "\n".join("  " + line for line in text.splitlines())


def analyze(
    returns: pd.Series[float],
    *,
    signal: pd.DataFrame | None = None,
    fwd_returns: pd.DataFrame | None = None,
    sample_end: str | pd.Timestamp | None = None,
    publication_date: str | pd.Timestamp | None = None,
    crowding: pd.DataFrame | pd.Series[float] | None = None,
    crowding_horizon: int | None = None,
    window: int | None = None,
    freq: str | None = None,
    ic_method: ICMethod = "spearman",
    min_assets: int = 5,
    n_boot: int = 1000,
    rng: RngLike = None,
) -> FadeReport:
    """Run every alphafade analysis on a strategy and bundle the results.

    Parameters
    ----------
    returns : Series
        Per-period long-short strategy returns (decimals), indexed by date.
    signal, fwd_returns : DataFrame, optional
        Dates x assets. Give both to add IC-based analysis. ``fwd_returns`` row t must hold
        returns realized after t (see :func:`forward_returns`).
    sample_end, publication_date : str or Timestamp, optional
        With both, runs :func:`publication_gap`. With only ``publication_date``, runs
        :func:`chow_test` at that date.
    crowding : DataFrame or Series, optional
        Output of :func:`crowding_score` (its ``mean`` column is used) or any crowding series
        on the same dates as ``returns``. Adds the crowding link: a Newey-West regression of
        the next ``crowding_horizon`` periods' compounded return on the crowding level.
    crowding_horizon : int, optional
        Periods ahead for the crowding link. Default: one year (12 months, 52 weeks, ...).
    window : int, optional
        Rolling window in periods. Default: three years (36 months, 156 weeks, 756 days).
    freq : str, optional
        'D', 'W', 'M', 'Q' or 'A'. Inferred if omitted.
    ic_method : {"spearman", "pearson"}, default "spearman"
    min_assets : int, default 5
    n_boot : int, default 1000
        Bootstrap replications for the decay fits.
    rng : int, numpy Generator, or None
        Seed for reproducible confidence intervals.

    Returns
    -------
    FadeReport
        Has ``.summary()``, ``.verdict()``, ``.to_frame()`` and ``.plot()``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> idx = pd.date_range("1965-01-31", "2015-12-31", freq="ME")
    >>> t = np.arange(len(idx)) / 12
    >>> r = pd.Series(0.012 * np.exp(-t / 15) + rng.normal(0, 0.01, len(idx)), index=idx)
    >>> report = analyze(r, sample_end="1989-12-31", publication_date="1993-03-01", rng=0)
    >>> report.return_decay.decay_detected
    True
    >>> print(report.verdict())  # doctest: +ELLIPSIS
    Yes: the strategy's average return is fading, with a half-life of about ... years...
    """
    r_full = as_series(returns, "returns")
    valid = r_full.notna().to_numpy()
    if not valid.any():
        raise InsufficientDataError("returns has no non-missing values.")
    first = int(np.argmax(valid))
    last = len(valid) - int(np.argmax(valid[::-1]))
    r = r_full.iloc[first:last]
    f = resolve_freq(pd.DatetimeIndex(r.index), freq, "returns")
    win = _DEFAULT_WINDOW[f] if window is None else window
    win = check_window(win, len(r))
    gen = resolve_rng(rng)

    sharpe = rolling_sharpe(r, window=win, freq=f)
    return_decay = fit_decay(r, n_boot=n_boot, rng=gen)
    brk = find_break(r)

    ic = ric = ic_decay = None
    if (signal is None) != (fwd_returns is None):
        raise InputError("Pass both signal and fwd_returns to include IC analysis, or neither.")
    if signal is not None and fwd_returns is not None:
        ic = ic_series(signal, fwd_returns, method=ic_method, min_assets=min_assets)
        ric = rolling_ic(
            signal, fwd_returns, window=min(win, len(ic)), method=ic_method, min_assets=min_assets
        )
        ic_decay = fit_decay(ic, n_boot=n_boot, rng=gen)

    gap = pub_test = None
    pub_ts = as_timestamp(publication_date, "publication_date") if publication_date else None
    end_ts = as_timestamp(sample_end, "sample_end") if sample_end else None
    if end_ts is not None and pub_ts is None:
        raise InputError("sample_end needs publication_date too (for the publication gap).")
    if pub_ts is not None and end_ts is not None:
        gap = publication_gap(r, sample_end=end_ts, publication_date=pub_ts, freq=f)
    elif pub_ts is not None:
        pub_test = chow_test(r, pub_ts)

    crowd_s = link = None
    if crowding is not None:
        crowd_s = _crowding_series(crowding)
        horizon = PERIODS_PER_YEAR[f] if crowding_horizon is None else crowding_horizon
        link = _crowding_link(r, crowd_s, horizon)

    return FadeReport(
        returns=r,
        freq=f,
        window=win,
        rolling_sharpe=sharpe,
        return_decay=return_decay,
        break_test=brk,
        ic=ic,
        rolling_ic=ric,
        ic_decay=ic_decay,
        publication=gap,
        publication_test=pub_test,
        publication_date=pub_ts,
        sample_end=end_ts,
        crowding=crowd_s,
        crowding_link=link,
    )


def _crowding_series(crowding: pd.DataFrame | pd.Series[float]) -> pd.Series[float]:
    if isinstance(crowding, pd.DataFrame):
        if "mean" not in crowding.columns:
            raise InputError(
                "crowding DataFrame must be crowding_score output (with a 'mean' column); "
                "otherwise pass a Series."
            )
        crowding = crowding["mean"]
    return as_series(crowding, "crowding").rename("crowding")


def _crowding_link(
    returns: pd.Series[float], crowding: pd.Series[float], horizon: int
) -> CrowdingLink:
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise InputError(f"crowding_horizon must be a positive integer, got {horizon!r}.")
    future = forward_returns(returns, periods=int(horizon)).rename("future")
    joined = pd.concat([crowding, future], axis=1, join="inner").dropna()
    n = len(joined)
    if n < 24:
        raise InsufficientDataError(
            f"Only {n} dates have both a crowding value and a {horizon}-period future return; "
            "need at least 24. Crowding must be on the same dates as returns (e.g. both "
            "month-end)."
        )
    c = to_array(joined["crowding"])
    y = to_array(joined["future"])
    sd = float(c.std(ddof=1))
    if not sd > 0:
        raise InputError("crowding is constant over the overlapping dates.")
    z = (c - c.mean()) / sd
    lags = resolve_hac_lags(None, n, min_lags=min(int(horizon) - 1, n - 1))
    res = ols_hac(y, np.column_stack([np.ones(n), z]), lags)
    return CrowdingLink(
        horizon=int(horizon),
        slope_per_sd=float(res.params[1]),
        t_stat=float(res.tvalues[1]),
        correlation=float(np.corrcoef(c, y)[0, 1]),
        n_obs=n,
    )
