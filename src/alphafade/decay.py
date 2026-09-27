"""Fit exponential (and linear) decay of a signal's edge across calendar time."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from ._errors import FitWarning, InputError, InsufficientDataError
from ._stats import (
    FloatArray,
    RngLike,
    block_bootstrap_indices,
    default_block_size,
    resolve_rng,
)
from ._validate import as_series, dropna_series, years_since_start
from .rolling import WINDOW_ATTR

__all__ = ["DecayFit", "fit_decay"]

Model = Literal["exponential", "linear"]

_MIN_OBS = 20
_GRID_POSITIVE = 160
_GRID_NEGATIVE = 60
_GOLDEN_ITERS = 30
_BOOT_CHUNK = 64


@dataclass(frozen=True)
class DecayFit:
    """Result of :func:`fit_decay`.

    Attributes
    ----------
    model : {"exponential", "linear"}
        Model behind the headline numbers. Exponential unless its fit failed, in which case
        alphafade fell back to linear and emitted a :class:`FitWarning`.
    decay_detected : bool
        True when the whole confidence interval for the decay rate is above zero.
    half_life_years : float or None
        Years for the edge to halve. None when no decay is detected, because a huge
        half-life from an insignificant fit would be meaningless. For the linear model this
        is the time from the start until the fitted line reaches half its starting level.
    ci_low, ci_high : float or None
        Confidence interval for the half-life. ``ci_high`` is ``inf`` when the interval for
        the rate includes zero (the data can't rule out "no decay").
    decay_rate : float
        Exponential: lambda in a * exp(-lambda * t), per year. Linear: the fall in the edge
        per year, as a share of the starting level. Positive means shrinking.
    rate_ci : tuple of float
        Bootstrap percentile interval for ``decay_rate``.
    p_value : float
        One-sided bootstrap p-value: the share of bootstrap fits showing no decay.
    initial_level : float
        Fitted edge at the first date (``a``, or the linear intercept).
    r2 : float
        R-squared of the headline model. On raw per-period data it is naturally small
        because single periods are noisy; that's expected.
    r2_exponential, r2_linear : float
    aic_exponential, aic_linear : float
        Akaike information criterion (lower is better). Both models have 2 parameters, so
        this is a fair comparison of fit.
    better_fit : {"exponential", "linear"}
    linear_slope : float
        Linear trend in the edge, in units per year.
    n_obs : int
    start, end : Timestamp
    block_size : int
        Block length (in observations) used by the bootstrap.
    alpha : float
        1 - confidence level.
    fitted : Series
        Headline model's fitted values.
    notes : tuple of str
        Caveats worth reading (e.g. the input was an overlapping rolling series).
    """

    model: Model
    decay_detected: bool
    half_life_years: float | None
    ci_low: float | None
    ci_high: float | None
    decay_rate: float
    rate_ci: tuple[float, float]
    p_value: float
    initial_level: float
    r2: float
    r2_exponential: float
    r2_linear: float
    aic_exponential: float
    aic_linear: float
    better_fit: Model
    linear_slope: float
    n_obs: int
    start: pd.Timestamp
    end: pd.Timestamp
    block_size: int
    alpha: float
    fitted: pd.Series[float] = field(repr=False)
    notes: tuple[str, ...] = ()

    def summary(self) -> str:
        """Plain-English description of the fit."""
        conf = f"{1 - self.alpha:.0%}"
        span = f"{self.start:%Y-%m} to {self.end:%Y-%m}"
        lines = [f"{self.model.capitalize()} decay fit on {self.n_obs} observations ({span})."]
        if self.decay_detected and self.half_life_years is not None:
            hi = "infinity" if self.ci_high is None or math.isinf(self.ci_high) else (
                f"{self.ci_high:.1f}"
            )
            lo = f"{self.ci_low:.1f}" if self.ci_low is not None else "?"
            if self.model == "exponential":
                per_year = 1 - math.exp(-self.decay_rate)
                lines.append(
                    f"The edge started at {self.initial_level:.4g} and is shrinking about "
                    f"{per_year:.1%} per year: half-life {self.half_life_years:.1f} years "
                    f"({conf} CI {lo} to {hi})."
                )
            else:
                lines.append(
                    f"The edge started at {self.initial_level:.4g} and is falling by "
                    f"{self.decay_rate:.1%} of that level per year, reaching half of it after "
                    f"{self.half_life_years:.1f} years ({conf} CI {lo} to {hi})."
                )
        else:
            lines.append(
                f"No detectable decay: the {conf} confidence interval for the decay rate "
                f"({self.rate_ci[0]:.3g} to {self.rate_ci[1]:.3g} per year) includes zero."
            )
            if self.rate_ci[1] < 0:
                lines.append("If anything, the edge has been growing.")
        diff = abs(self.aic_exponential - self.aic_linear)
        other = "linear" if self.better_fit == "exponential" else "exponential"
        lines.append(
            f"The {self.better_fit} model fits slightly better than the {other} one "
            f"(AIC {diff:.1f} lower)."
            if diff < 2
            else f"The {self.better_fit} model fits better than the {other} one "
            f"(AIC {diff:.1f} lower)."
        )
        lines.extend(f"Note: {n}" for n in self.notes)
        return "\n".join(lines)


def fit_decay(
    perf: pd.Series[float],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    block_size: int | None = None,
    rng: RngLike = None,
) -> DecayFit:
    """Fit how a signal's edge decays over calendar time and report its half-life.

    Fits p(t) = a * exp(-lambda * t), where t is years since the first date, and also a
    straight line p(t) = b0 + b1 * t for comparison. The half-life is ln(2) / lambda. The
    confidence interval comes from a residual moving-block bootstrap: resample blocks of
    residuals (keeping the time axis fixed so the trend isn't scrambled), refit, repeat.
    Blocks keep the autocorrelation that raw financial series usually have.

    Parameters
    ----------
    perf : Series
        Performance over time, indexed by date. Best: a raw per-period series such as
        :func:`ic_series` output or periodic strategy returns. Rolling series
        (:func:`rolling_ic`, :func:`rolling_sharpe`) also work, but their overlapping windows
        inflate R-squared; alphafade widens the bootstrap blocks to the window length and
        adds a note.
    n_boot : int, default 1000
        Bootstrap replications.
    alpha : float, default 0.05
        Significance level; the confidence intervals cover 1 - alpha.
    block_size : int, optional
        Bootstrap block length in observations. Default: about 1.75 * n^(1/3), and never
        shorter than the rolling window when ``perf`` is a rolling series.
    rng : int, numpy Generator, or None
        Seed for reproducible confidence intervals. Pass an int.

    Returns
    -------
    DecayFit

    Warns
    -----
    FitWarning
        If the exponential fit fails (the decay is faster or the growth steeper than the data
        can resolve) and the linear model is used instead.
    DataDroppedWarning
        If ``perf`` has missing values in the middle.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> dates = pd.date_range("1970-01-31", periods=480, freq="ME")
    >>> t = np.arange(480) / 12
    >>> noise = np.random.default_rng(0).normal(0, 0.02, 480)
    >>> ic = pd.Series(0.10 * np.exp(-t / 5) + noise, index=dates)
    >>> fit = fit_decay(ic, n_boot=200, rng=0)
    >>> fit.decay_detected, round(fit.half_life_years, 1)
    (True, 3.5)
    """
    series = as_series(perf, "perf")
    window = series.attrs.get(WINDOW_ATTR)
    y_s = dropna_series(series, "perf")
    n = len(y_s)
    if n < _MIN_OBS:
        raise InsufficientDataError(
            f"fit_decay needs at least {_MIN_OBS} non-missing observations, got {n}."
        )
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 100:
        raise InputError(f"n_boot must be an integer of at least 100, got {n_boot!r}.")
    if not 0 < alpha < 0.5:
        raise InputError(f"alpha must be between 0 and 0.5, got {alpha!r}.")
    y = y_s.to_numpy(dtype=np.float64)
    t = years_since_start(pd.DatetimeIndex(y_s.index))
    span = float(t[-1])
    sst = float(((y - y.mean()) ** 2).sum())
    if sst <= 1e-300:
        raise InputError("perf is constant, so there is no decay (or anything else) to fit.")

    notes: list[str] = []
    min_block = 1
    if isinstance(window, (int, np.integer)) and window > 1:
        min_block = int(window)
        notes.append(
            f"perf is a rolling series (window={window}). Neighbouring values overlap, so "
            "R-squared is inflated; bootstrap blocks were widened to the window length. "
            "For cleaner inference fit the raw per-period series instead."
        )
    if block_size is None:
        block = default_block_size(n, min_block=min_block)
    else:
        if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)):
            raise InputError(f"block_size must be a positive integer, got {block_size!r}.")
        if not 1 <= block_size <= n:
            raise InputError(f"block_size must be between 1 and {n}, got {block_size}.")
        block = int(block_size)
    if n / block < 10:
        notes.append(
            f"Only about {n // block} independent blocks of data; the confidence interval "
            "is rough. More history would sharpen it."
        )

    grid = _lambda_grid(t)
    lam, a, sse_exp = _fit_exponential(t, y[None, :], grid)
    lam0, a0, sse_e = float(lam[0]), float(a[0]), float(sse_exp[0])
    b0, b1, sse_lin = _fit_linear(t, y)

    aic_e = _aic(sse_e, n)
    aic_l = _aic(sse_lin, n)
    better: Model = "exponential" if aic_e <= aic_l else "linear"

    exp_failed = lam0 <= grid[0] or lam0 >= grid[-1] or not math.isfinite(a0)
    rng_g = resolve_rng(rng)
    if exp_failed:
        how = "faster" if lam0 >= grid[-1] else "steeper growth"
        warnings.warn(
            f"The exponential fit hit its bound ({how} than {span:.1f} years of data can "
            "resolve), so the linear model is used for the headline numbers.",
            FitWarning,
            stacklevel=2,
        )
        notes.append("Exponential fit failed; using the linear trend instead.")
        model: Model = "linear"
        fitted = b0 + b1 * t
        boot = _boot_linear(t, fitted, y - fitted, block, n_boot, rng_g)
        sign = 1.0 if b0 >= 0 else -1.0
        rate = -b1 * sign / abs(b0) if b0 != 0 else 0.0
        rates = -boot * sign / abs(b0) if b0 != 0 else np.zeros_like(boot)
        initial = b0
        r2 = 1 - sse_lin / sst
    else:
        model = "exponential"
        fitted = a0 * np.exp(-lam0 * t)
        rates = _boot_exponential(t, fitted, y - fitted, block, n_boot, grid, rng_g)
        rate = lam0
        initial = a0
        r2 = 1 - sse_e / sst

    lo_q, hi_q = np.quantile(rates, [alpha / 2, 1 - alpha / 2])
    rate_ci = (float(lo_q), float(hi_q))
    p_value = float(np.mean(rates <= 0))
    detected = bool(rate_ci[0] > 0 and rate > 0)
    if detected:
        half = math.log(2) / rate
        ci_low = math.log(2) / rate_ci[1]
        ci_high = math.log(2) / rate_ci[0]
    else:
        half = ci_low = ci_high = None
    if detected and model == "linear":
        # Linear "half-life": years until the line reaches half of its starting level.
        half = 0.5 / rate
        ci_low = 0.5 / rate_ci[1]
        ci_high = 0.5 / rate_ci[0]

    return DecayFit(
        model=model,
        decay_detected=detected,
        half_life_years=half,
        ci_low=ci_low,
        ci_high=ci_high,
        decay_rate=float(rate),
        rate_ci=rate_ci,
        p_value=p_value,
        initial_level=float(initial),
        r2=float(r2),
        r2_exponential=float(1 - sse_e / sst),
        r2_linear=float(1 - sse_lin / sst),
        aic_exponential=aic_e,
        aic_linear=aic_l,
        better_fit=better,
        linear_slope=float(b1),
        n_obs=n,
        start=pd.Timestamp(y_s.index[0]),
        end=pd.Timestamp(y_s.index[-1]),
        block_size=block,
        alpha=float(alpha),
        fitted=pd.Series(fitted, index=y_s.index, name=f"{model}_fit"),
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------------------
# Model fitting internals
# --------------------------------------------------------------------------------------
def _lambda_grid(t: FloatArray) -> FloatArray:
    """Grid of candidate decay rates (per year).

    Upper bound: a half-life of 1% of the sample span (or 3 observations, if longer).
    Lower bound: growth of at most 100x over the sample. Points are log-spaced on each side
    of zero so both slow and fast decay are resolved, and 0 (no decay) is always included.
    """
    span = float(t[-1])
    min_step = float(np.median(np.diff(t)))
    min_half_life = max(span / 100.0, 3.0 * min_step)
    lam_max = math.log(2) / min_half_life
    lam_min = -math.log(100.0) / span
    pos = np.geomspace(lam_max * 1e-5, lam_max, _GRID_POSITIVE)
    neg = -np.geomspace(-lam_min * 1e-5, -lam_min, _GRID_NEGATIVE)[::-1]
    return np.concatenate([neg, [0.0], pos])


def _profile(t: FloatArray, y: FloatArray, lam: FloatArray) -> tuple[FloatArray, FloatArray]:
    """For each row of ``y`` and its rate in ``lam``, return the best ``a`` and the SSE.

    For a fixed rate the best level has a closed form (a least-squares projection), so the
    exponential fit reduces to a 1-D search over the rate. This can't diverge.
    """
    e = np.exp(-lam[:, None] * t[None, :])
    s1 = np.einsum("ij,ij->i", y, e)
    s2 = np.einsum("ij,ij->i", e, e)
    a = s1 / s2
    sse = np.einsum("ij,ij->i", y, y) - s1 * a
    return a, sse


def _fit_exponential(
    t: FloatArray, y: FloatArray, grid: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Vectorized fit for every row of ``y``: grid search, then golden-section refinement."""
    e = np.exp(-np.outer(grid, t))  # (G, n)
    s1 = y @ e.T  # (B, G)
    s2 = np.einsum("ij,ij->i", e, e)  # (G,)
    sse_grid = np.einsum("ij,ij->i", y, y)[:, None] - s1**2 / s2[None, :]
    best = np.argmin(sse_grid, axis=1)
    lo = grid[np.maximum(best - 1, 0)]
    hi = grid[np.minimum(best + 1, len(grid) - 1)]
    inv_phi = (math.sqrt(5) - 1) / 2
    c = hi - inv_phi * (hi - lo)
    d = lo + inv_phi * (hi - lo)
    _, fc = _profile(t, y, c)
    _, fd = _profile(t, y, d)
    for _ in range(_GOLDEN_ITERS):
        left = fc < fd
        hi = np.where(left, d, hi)
        lo = np.where(left, lo, c)
        new_c = hi - inv_phi * (hi - lo)
        new_d = lo + inv_phi * (hi - lo)
        c_next = np.where(left, new_c, d)
        d_next = np.where(left, c, new_d)
        fc_next = np.where(left, np.nan, fd)
        fd_next = np.where(left, fc, np.nan)
        need_c = np.isnan(fc_next)
        need_d = np.isnan(fd_next)
        if need_c.any():
            fc_next[need_c] = _profile(t, y[need_c], c_next[need_c])[1]
        if need_d.any():
            fd_next[need_d] = _profile(t, y[need_d], d_next[need_d])[1]
        c, d, fc, fd = c_next, d_next, fc_next, fd_next
    lam = (lo + hi) / 2
    # Keep the exact grid endpoint when the optimum is at the boundary, so callers can
    # detect "hit the bound".
    at_edge = (best == 0) | (best == len(grid) - 1)
    lam = np.where(at_edge, grid[best], lam)
    a, sse = _profile(t, y, lam)
    grid_best = sse_grid[np.arange(len(best)), best]
    worse = sse > grid_best
    if worse.any():  # golden search can't beat the grid point it started from: keep it
        lam = np.where(worse, grid[best], lam)
        a, sse = _profile(t, y, lam)
    return lam, a, sse


def _fit_linear(t: FloatArray, y: FloatArray) -> tuple[float, float, float]:
    tc = t - t.mean()
    b1 = float((tc * (y - y.mean())).sum() / (tc * tc).sum())
    b0 = float(y.mean() - b1 * t.mean())
    sse = float(((y - b0 - b1 * t) ** 2).sum())
    return b0, b1, sse


def _aic(sse: float, n: int, k: int = 2) -> float:
    return float(n * math.log(max(sse, 1e-300) / n) + 2 * k)


def _boot_samples(
    fitted: FloatArray,
    resid: FloatArray,
    block: int,
    n_boot: int,
    rng: np.random.Generator,
) -> FloatArray:
    n = len(fitted)
    centered = resid - resid.mean()
    idx = np.stack([block_bootstrap_indices(n, block, rng) for _ in range(n_boot)])
    out: FloatArray = fitted[None, :] + centered[idx]
    return out


def _boot_exponential(
    t: FloatArray,
    fitted: FloatArray,
    resid: FloatArray,
    block: int,
    n_boot: int,
    grid: FloatArray,
    rng: np.random.Generator,
) -> FloatArray:
    rates = np.empty(n_boot)
    for start in range(0, n_boot, _BOOT_CHUNK):
        size = min(_BOOT_CHUNK, n_boot - start)
        ys = _boot_samples(fitted, resid, block, size, rng)
        rates[start : start + size] = _fit_exponential(t, ys, grid)[0]
    return rates


def _boot_linear(
    t: FloatArray,
    fitted: FloatArray,
    resid: FloatArray,
    block: int,
    n_boot: int,
    rng: np.random.Generator,
) -> FloatArray:
    tc = t - t.mean()
    slopes = np.empty(n_boot)
    for start in range(0, n_boot, _BOOT_CHUNK):
        size = min(_BOOT_CHUNK, n_boot - start)
        ys = _boot_samples(fitted, resid, block, size, rng)
        slopes[start : start + size] = (ys - ys.mean(axis=1, keepdims=True)) @ tc / (tc @ tc)
    return slopes
