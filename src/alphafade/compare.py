"""Compare decay across many signals, correcting for the fact that many were tested."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from ._errors import DataDroppedWarning, FitWarning, InputError, InsufficientDataError
from ._stats import FloatArray, RngLike
from ._validate import as_panel, dropna_series
from .decay import DecayFit, fit_decay

__all__ = ["SignalComparison", "compare_signals"]

Adjust = Literal["holm", "bh", "none"]

_MIN_OBS = 20
_ADJUST_NAMES = {
    "holm": "Holm step-down (controls the chance of even one false fade)",
    "bh": "Benjamini-Hochberg (controls the share of flagged fades that are false)",
    "none": "no correction",
}


def _adjust_pvalues(p: FloatArray, method: Adjust) -> FloatArray:
    """Multiple-testing adjustment of a vector of p-values.

    Holm: sort ascending, multiply the i-th smallest (0-based) by ``m - i``, then take a
    running maximum so adjusted values never fall as raw p rises. BH: sort descending,
    multiply the value of rank ``r`` by ``m / r``, then take a running minimum. Both are
    capped at 1 and returned in the original order.
    """
    p = np.asarray(p, dtype=np.float64)
    m = len(p)
    if method == "none" or m == 0:
        return p.copy()
    order = np.argsort(p, kind="stable")
    ps = p[order]
    if method == "holm":
        adj = np.maximum.accumulate(ps * (m - np.arange(m)))
    elif method == "bh":
        scaled = ps * m / np.arange(1, m + 1)
        adj = np.minimum.accumulate(scaled[::-1])[::-1]
    else:  # pragma: no cover - guarded by the public function
        raise InputError(f"adjust must be 'holm', 'bh' or 'none', got {method!r}.")
    out = np.empty(m, dtype=np.float64)
    out[order] = np.minimum(adj, 1.0)
    return out


@dataclass(frozen=True)
class SignalComparison:
    """Result of :func:`compare_signals`.

    Attributes
    ----------
    table : DataFrame
        One row per usable signal, ranked fastest-fading first. Columns: ``decay_rate``,
        ``half_life_years`` (NaN when no decay is detected), ``ci_low``, ``ci_high``,
        ``p_value`` (raw one-sided bootstrap p), ``p_adjusted`` (after the multiple-testing
        correction), ``decay_detected`` (the single-signal verdict),
        ``decay_detected_adjusted`` (the verdict that survives the correction: adjusted
        p below ``alpha`` AND the single-signal verdict), ``n_obs``, ``start``, ``end``,
        ``model`` and ``rank``.
    fits : dict of str to DecayFit
        The full fit for each usable signal.
    adjust : {"holm", "bh", "none"}
        The correction that was applied.
    alpha : float
        Significance level used for the fits and for the adjusted p-values.
    skipped : tuple of str
        Names of columns that were left out because they had too little usable data.
    """

    table: pd.DataFrame
    fits: dict[str, DecayFit] = field(repr=False)
    adjust: Adjust
    alpha: float
    skipped: tuple[str, ...] = ()

    @property
    def n_detected_adjusted(self) -> int:
        """Number of signals whose decay survives the multiple-testing correction."""
        return int(self.table["decay_detected_adjusted"].sum())

    def summary(self) -> str:
        """Plain-English description of the comparison."""
        t = self.table
        n = len(t)
        raw = int(t["decay_detected"].sum())
        kept = self.n_detected_adjusted
        lines = [
            f"Compared {n} signals. {raw} show decay on their own; after "
            f"{_ADJUST_NAMES[self.adjust]}, {kept} still do."
        ]
        if self.skipped:
            lines.append(f"Skipped for too little data: {', '.join(self.skipped)}.")
        confirmed = t[t["decay_detected_adjusted"]]
        if len(confirmed):
            fastest = confirmed.index[0]
            hl = float(confirmed.loc[fastest, "half_life_years"])
            lines.append(f"Fastest confirmed fade: {fastest}, half-life {hl:.1f} years.")
        else:
            lines.append("No signal shows decay that survives the correction.")
        if self.adjust == "none":
            lines.append(
                "Warning: with no correction, testing many signals will flag fades by pure "
                "luck. At a 5% level, about 1 in 20 pure-noise signals looks like it is "
                "fading."
            )
        else:
            lines.append(
                "Why correct: each signal has a small chance of looking like it is fading "
                "by luck alone, and testing many signals multiplies the chances that at "
                "least one does. The correction raises each p-value to account for how "
                "many signals were tested, which removes most, though not all, false alarms."
            )
        return "\n".join(lines)


def compare_signals(
    perf: pd.DataFrame,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    adjust: Adjust = "holm",
    rng: RngLike = None,
) -> SignalComparison:
    """Fit decay for many signals at once and correct for multiple testing.

    Runs :func:`fit_decay` on every column, then adjusts the p-values. The adjustment
    matters because when you test many signals, some will look like they are fading purely
    by luck. Holm controls the family-wise error rate (the chance of even one false fade
    among all signals); Benjamini-Hochberg controls the false discovery rate (the expected
    share of flagged fades that are false) and is more lenient.

    Parameters
    ----------
    perf : DataFrame
        Dates by signals: each column is a strategy-return or IC series. Columns may start
        and end on different dates; leading and trailing NaNs are trimmed per column.
    n_boot : int, default 1000
        Bootstrap replications per signal.
    alpha : float, default 0.05
        Significance level for the confidence intervals and the adjusted p-values.
    adjust : {"holm", "bh", "none"}, default "holm"
        Multiple-testing correction.
    rng : int, numpy Generator, or None
        Seed. Each column gets its own independent sub-seed derived from this seed and the
        column's name, so the same seed reproduces results exactly, whatever the column
        order and whichever other columns are present or skipped.

    Returns
    -------
    SignalComparison

    Raises
    ------
    InputError
        For bad arguments, or a constant column-set leaving fewer than 2 usable signals.
    InsufficientDataError
        If fewer than 2 columns have enough data.

    Warns
    -----
    DataDroppedWarning
        For columns skipped for having too little data, and for interior missing values.
    FitWarning
        Once, naming the signals whose exponential fit failed and fell back to linear.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> dates = pd.date_range("1970-01-31", periods=480, freq="ME")
    >>> t = np.arange(480) / 12
    >>> g = np.random.default_rng(2)
    >>> df = pd.DataFrame({
    ...     "fading": 0.10 * np.exp(-t / 4) + g.normal(0, 0.02, 480),
    ...     "noise": g.normal(0, 0.02, 480),
    ... }, index=dates)
    >>> res = compare_signals(df, n_boot=200, rng=0)
    >>> list(res.table.index), res.table["decay_detected_adjusted"].tolist()
    (['fading', 'noise'], [True, False])
    """
    panel = as_panel(perf, "perf")
    if adjust not in ("holm", "bh", "none"):
        raise InputError(f"adjust must be 'holm', 'bh' or 'none', got {adjust!r}.")
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 100:
        raise InputError(f"n_boot must be an integer of at least 100, got {n_boot!r}.")
    if not 0 < alpha < 0.5:
        raise InputError(f"alpha must be between 0 and 0.5, got {alpha!r}.")

    if isinstance(rng, np.random.Generator):
        root = np.random.SeedSequence(rng.integers(0, 2**63, size=4).tolist())
    elif rng is None or (isinstance(rng, (int, np.integer)) and not isinstance(rng, bool)):
        root = np.random.SeedSequence(None if rng is None else int(rng))
    else:
        raise InputError(f"rng must be an int seed, a numpy Generator, or None; got {rng!r}.")
    # Each column's random stream is keyed by its NAME (not its position), so reordering,
    # adding, or skipping other columns never changes a column's result.
    children = [
        np.random.SeedSequence(root.entropy, spawn_key=(_name_key(col),)) for col in panel.columns
    ]

    fits: dict[str, DecayFit] = {}
    skipped: list[str] = []
    fallback: list[str] = []
    for pos, col in enumerate(panel.columns):
        name = str(col)
        series = panel.iloc[:, pos]
        if not series.notna().any():
            skipped.append(name)
            continue
        clean = dropna_series(series, f"perf[{name!r}]")
        if len(clean) < _MIN_OBS or np.ptp(clean.to_numpy()) == 0:
            skipped.append(name)
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", FitWarning)
            fits[name] = fit_decay(
                clean, n_boot=n_boot, alpha=alpha, rng=np.random.default_rng(children[pos])
            )
        if any(issubclass(w.category, FitWarning) for w in caught):
            fallback.append(name)
        for w in caught:
            if not issubclass(w.category, FitWarning):
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    if skipped:
        warnings.warn(
            f"Skipped {len(skipped)} signal(s) with fewer than {_MIN_OBS} usable "
            f"observations or no variation: {', '.join(skipped)}.",
            DataDroppedWarning,
            stacklevel=2,
        )
    if fallback:
        warnings.warn(
            "The exponential fit hit its bound, so the linear model was used for: "
            f"{', '.join(fallback)}. Treat their half-lives with extra care.",
            FitWarning,
            stacklevel=2,
        )
    if len(fits) < 2:
        raise InsufficientDataError(
            f"compare_signals needs at least 2 usable signals, got {len(fits)} "
            f"(skipped: {skipped or 'none'})."
        )

    names = list(fits)
    raw_p = np.array([fits[k].p_value for k in names], dtype=np.float64)
    adj_p = _adjust_pvalues(raw_p, adjust)
    rows = []
    for k, pa in zip(names, adj_p, strict=True):
        f = fits[k]
        rows.append(
            {
                "signal": k,
                "decay_rate": f.decay_rate,
                "half_life_years": np.nan if f.half_life_years is None else f.half_life_years,
                "ci_low": np.nan if f.ci_low is None else f.ci_low,
                "ci_high": np.nan if f.ci_high is None else f.ci_high,
                "p_value": f.p_value,
                "p_adjusted": float(pa),
                "decay_detected": bool(f.decay_detected),
                "decay_detected_adjusted": bool(pa < alpha and f.decay_detected),
                "n_obs": f.n_obs,
                "start": f.start,
                "end": f.end,
                "model": f.model,
            }
        )
    df = pd.DataFrame(rows).set_index("signal")
    df.index.name = None
    # Detected decay first (shortest half-life first), then the rest by raw p-value.
    detected = df["decay_detected"]
    key = pd.DataFrame(
        {
            "grp": (~detected).astype(int),
            "a": df["half_life_years"].where(detected, df["p_value"]),
        }
    )
    order = key.reset_index(drop=True).sort_values(["grp", "a"], kind="stable").index
    df = df.iloc[list(order)]
    df["rank"] = np.arange(1, len(df) + 1)
    return SignalComparison(
        table=df, fits=fits, adjust=adjust, alpha=alpha, skipped=tuple(skipped)
    )


def _name_key(name: object) -> int:
    """Return a stable 64-bit integer for a column label (same across runs and machines)."""
    digest = hashlib.blake2b(repr(name).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")
