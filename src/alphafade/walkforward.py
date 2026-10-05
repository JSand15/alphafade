"""Walk-forward decay: refit the half-life on expanding windows to see if it is stable."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ._errors import FitWarning, InputError, InsufficientDataError
from ._stats import RngLike
from ._validate import as_series, dropna_series
from .decay import fit_decay

__all__ = ["WalkForwardResult", "walk_forward_decay"]

_MIN_OBS = 20
_COLUMNS = [
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


@dataclass(frozen=True)
class WalkForwardResult:
    """Result of :func:`walk_forward_decay`.

    Attributes
    ----------
    table : DataFrame
        One row per window, indexed by the window's end date. Columns: ``n_obs``,
        ``decay_rate``, ``half_life_years`` (NaN when no decay is detected), ``ci_low``,
        ``ci_high``, ``p_value``, ``decay_detected``, ``model`` and ``fallback`` (True when
        the exponential fit failed and the linear model was used).
    share_detected : float
        Share of windows in which decay was detected.
    stable_since : Timestamp or None
        First window end date from which decay was detected in that window and in every
        later one. None if decay is not detected in the final window.
    half_life_drift : float or None
        Relative spread of the detected half-lives, ``(max - min) / median``. None with
        fewer than two detected windows. Near 0 means a steady estimate; above about 1
        means the estimate moves by more than its own size.
    min_obs, step : int
    alpha : float
    """

    table: pd.DataFrame = field(repr=False)
    share_detected: float
    stable_since: pd.Timestamp | None
    half_life_drift: float | None
    min_obs: int
    step: int
    alpha: float

    @property
    def stability(self) -> dict[str, Any]:
        """The stability measures as a plain dict."""
        return {
            "share_detected": self.share_detected,
            "stable_since": self.stable_since,
            "half_life_drift": self.half_life_drift,
        }

    def summary(self) -> str:
        """Plain-English description of how stable the estimate has been."""
        n = len(self.table)
        detected = int(self.table["decay_detected"].sum())
        first, last = self.table.index[0], self.table.index[-1]
        lines = [
            f"Walk-forward decay: {n} expanding windows ending {first:%Y-%m} to {last:%Y-%m}, "
            f"each using only data available at its end date.",
            f"Decay was detected in {detected} of {n} windows ({self.share_detected:.0%}).",
        ]
        if self.stable_since is not None:
            lines.append(f"Detection has held in every window since {self.stable_since:%Y-%m}.")
        elif detected == 0:
            lines.append("No window detected decay.")
        else:
            lines.append(
                "The latest (full-sample) window does not detect decay, so there is no "
                "full-sample half-life to trust; the earlier detections did not last."
            )
        if self.half_life_drift is not None:
            hl = self.table["half_life_years"].dropna()
            lines.append(
                f"Detected half-lives ranged from {hl.min():.1f} to {hl.max():.1f} years "
                f"(spread {self.half_life_drift:.0%} of the median)."
            )
            if self.half_life_drift > 0.5:
                lines.append(
                    "The estimate keeps changing as data is added, so a single full-sample "
                    "half-life should not be trusted: it depends heavily on the sample end "
                    "date, which suggests the decay is not a stable process."
                )
            elif self.stable_since is not None:
                lines.append(
                    "The estimate is fairly steady as data is added, which supports "
                    "trusting a full-sample half-life."
                )
        else:
            lines.append("Fewer than two windows detected decay, so no half-life drift is given.")
        if self.table["fallback"].any():
            lines.append(
                f"{int(self.table['fallback'].sum())} window(s) used the linear fallback "
                "because the exponential fit failed."
            )
        return "\n".join(lines)


def _window_ends(n: int, min_obs: int, step: int) -> list[int]:
    ends = list(range(min_obs, n + 1, step))
    if ends[-1] != n:
        ends.append(n)
    return ends


def _seed_sequence(rng: RngLike) -> np.random.SeedSequence:
    if isinstance(rng, np.random.Generator):
        return np.random.SeedSequence(int(rng.integers(0, 2**63 - 1)))
    if rng is None or (isinstance(rng, (int, np.integer)) and not isinstance(rng, bool)):
        return np.random.SeedSequence(rng)
    raise InputError(f"rng must be an int seed, a numpy Generator, or None; got {rng!r}.")


def walk_forward_decay(
    perf: pd.Series[float],
    *,
    min_obs: int = 60,
    step: int = 12,
    n_boot: int = 200,
    alpha: float = 0.05,
    rng: RngLike = None,
) -> WalkForwardResult:
    """Refit :func:`fit_decay` on expanding windows to test whether the half-life is stable.

    A single full-sample half-life hides whether the answer was always about the same or
    swung around as history accumulated. This refits on ``perf[:k]`` for ``k = min_obs,
    min_obs + step, ...`` and finally the full sample. Each window uses only data up to its
    end date, so there is no look-ahead (a row never changes if later data changes). Each
    window gets its own bootstrap seed spawned from ``rng``, so results are reproducible.

    Parameters
    ----------
    perf : Series
        Performance over time, indexed by date (see :func:`fit_decay`).
    min_obs : int, default 60
        Observations in the first window; at least 20.
    step : int, default 12
        Observations added per window.
    n_boot : int, default 200
        Bootstrap replications per window (at least 100).
    alpha : float, default 0.05
        Significance level for each window's confidence intervals.
    rng : int, numpy Generator, or None
        Seed. Pass an int for reproducible output.

    Returns
    -------
    WalkForwardResult

    Raises
    ------
    InputError
        For bad arguments or a constant series.
    InsufficientDataError
        If there are fewer than ``min_obs`` non-missing observations.

    Warns
    -----
    DataDroppedWarning
        If ``perf`` has missing values in the middle.

    Notes
    -----
    A window whose exponential fit fails falls back to the linear model; this is recorded
    in the ``model`` and ``fallback`` columns instead of raising or warning.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> dates = pd.date_range("1970-01-31", periods=360, freq="ME")
    >>> t = np.arange(360) / 12
    >>> noise = np.random.default_rng(0).normal(0, 0.01, 360)
    >>> ic = pd.Series(0.10 * np.exp(-t / 5) + noise, index=dates)
    >>> res = walk_forward_decay(ic, min_obs=120, step=120, n_boot=100, rng=0)
    >>> res.table["n_obs"].tolist()
    [120, 240, 360]
    """
    series = as_series(perf, "perf")
    if (
        isinstance(min_obs, bool)
        or not isinstance(min_obs, (int, np.integer))
        or min_obs < _MIN_OBS
    ):
        raise InputError(f"min_obs must be an integer of at least {_MIN_OBS}, got {min_obs!r}.")
    if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 1:
        raise InputError(f"step must be a positive integer, got {step!r}.")
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 100:
        raise InputError(f"n_boot must be an integer of at least 100, got {n_boot!r}.")
    if not 0 < alpha < 0.5:
        raise InputError(f"alpha must be between 0 and 0.5, got {alpha!r}.")
    clean = dropna_series(series, "perf")
    clean.attrs = dict(series.attrs)
    n = len(clean)
    if n < min_obs:
        raise InsufficientDataError(
            f"walk_forward_decay needs at least min_obs={min_obs} non-missing observations "
            f"for one window, got {n}."
        )
    if np.ptp(clean.to_numpy()) == 0:
        raise InputError("perf is constant, so there is no decay (or anything else) to fit.")

    ends = _window_ends(n, int(min_obs), int(step))
    seeds = _seed_sequence(rng).spawn(len(ends))
    rows: list[dict[str, Any]] = []
    for k, seed in zip(ends, seeds, strict=True):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = fit_decay(
                clean.iloc[:k],
                n_boot=int(n_boot),
                alpha=alpha,
                rng=np.random.default_rng(seed),
            )
        for w in caught:
            if not issubclass(w.category, FitWarning):
                warnings.warn_explicit(
                    w.message, w.category, w.filename, w.lineno, source=w.source
                )
        rows.append(
            {
                "n_obs": fit.n_obs,
                "decay_rate": fit.decay_rate,
                "half_life_years": np.nan if fit.half_life_years is None else fit.half_life_years,
                "ci_low": np.nan if fit.ci_low is None else fit.ci_low,
                "ci_high": np.nan if fit.ci_high is None else fit.ci_high,
                "p_value": fit.p_value,
                "decay_detected": fit.decay_detected,
                "model": fit.model,
                "fallback": any(issubclass(w.category, FitWarning) for w in caught),
            }
        )
    table = pd.DataFrame(rows, columns=_COLUMNS, index=clean.index[[k - 1 for k in ends]])
    table.index.name = None
    table["decay_detected"] = table["decay_detected"].astype(bool)
    table["fallback"] = table["fallback"].astype(bool)

    det = table["decay_detected"].to_numpy()
    stable: pd.Timestamp | None = None
    if det[-1]:
        first_bad = int(np.max(np.nonzero(~det)[0])) + 1 if (~det).any() else 0
        stable = pd.Timestamp(table.index[first_bad])
    hl = table["half_life_years"].dropna()
    drift: float | None = None
    if len(hl) >= 2 and hl.median() > 0:
        drift = float((hl.max() - hl.min()) / hl.median())
    return WalkForwardResult(
        table=table,
        share_detected=float(det.mean()),
        stable_since=stable,
        half_life_drift=drift,
        min_obs=int(min_obs),
        step=int(step),
        alpha=float(alpha),
    )
