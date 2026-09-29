"""Small statistical building blocks: Newey-West OLS, lag rules, block bootstrap, RNG.

These are deliberately plain numpy so the core install stays small. The Newey-West code is
checked against statsmodels in the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._errors import InputError, InsufficientDataError

FloatArray = NDArray[np.float64]
RngLike = int | np.random.Generator | None


def resolve_rng(rng: RngLike) -> np.random.Generator:
    """Turn a seed, Generator, or None into a Generator."""
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None or (isinstance(rng, (int, np.integer)) and not isinstance(rng, bool)):
        return np.random.default_rng(rng)
    raise InputError(f"rng must be an int seed, a numpy Generator, or None; got {rng!r}.")


def default_hac_lags(n: int) -> int:
    """Newey-West (1994) plug-in lag rule: floor(4 * (n / 100) ** (2 / 9))."""
    return int(4.0 * (n / 100.0) ** (2.0 / 9.0))  # truncation == floor for n > 0


def resolve_hac_lags(hac_lags: int | None, n: int, min_lags: int = 0) -> int:
    """Validate a user lag choice, or apply the default (never below ``min_lags``).

    ``min_lags`` only floors the *default*. An explicit ``hac_lags`` is used as given, so the
    caller stays in control (and owns the consequences of too few lags).
    """
    if hac_lags is None:
        return max(default_hac_lags(n), min_lags)
    if isinstance(hac_lags, bool) or not isinstance(hac_lags, (int, np.integer)):
        raise InputError(f"hac_lags must be a non-negative integer, got {hac_lags!r}.")
    if hac_lags < 0:
        raise InputError(f"hac_lags must be non-negative, got {hac_lags}.")
    if hac_lags >= n:
        raise InputError(f"hac_lags={hac_lags} must be smaller than the sample size ({n}).")
    return int(hac_lags)


@dataclass(frozen=True)
class OLSResult:
    """Coefficients and Newey-West standard errors from :func:`ols_hac`."""

    params: FloatArray
    bse: FloatArray
    cov: FloatArray
    resid: FloatArray
    nobs: int
    lags: int

    @property
    def tvalues(self) -> FloatArray:
        """Coefficient / standard error (inf-safe: zero SE gives nan)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            t: FloatArray = np.where(self.bse > 0, self.params / self.bse, np.nan)
        return t


def ols_hac(y: FloatArray, x: FloatArray, lags: int) -> OLSResult:
    """OLS with Newey-West (Bartlett kernel) HAC covariance.

    Matches ``statsmodels.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})``
    (no small-sample degrees-of-freedom correction, as in Newey & West 1987).

    Parameters
    ----------
    y : ndarray, shape (n,)
    x : ndarray, shape (n, k)
        Include a column of ones for an intercept.
    lags : int
        Number of autocovariance lags; 0 gives White (heteroskedasticity-only) errors.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n, k = x.shape
    if n <= k:
        raise InsufficientDataError(
            f"Need more observations ({n}) than regression coefficients ({k})."
        )
    xtx = x.T @ x
    if np.linalg.matrix_rank(xtx) < k:
        raise InsufficientDataError(
            "The regression is singular (a regressor is constant or duplicated). This "
            "usually means one period or group has no observations."
        )
    xtx_inv = np.linalg.inv(xtx)
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta
    scores = x * resid[:, None]
    s = scores.T @ scores
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        s += w * (gamma + gamma.T)
    cov = xtx_inv @ s @ xtx_inv
    bse = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return OLSResult(params=beta, bse=bse, cov=cov, resid=resid, nobs=n, lags=lags)


def hac_mean(y: FloatArray, lags: int) -> tuple[float, float, float]:
    """Mean of ``y`` with its Newey-West standard error and t-stat."""
    res = ols_hac(y, np.ones((len(y), 1)), lags)
    return float(res.params[0]), float(res.bse[0]), float(res.tvalues[0])


def default_block_size(n: int, min_block: int = 1) -> int:
    """Moving-block bootstrap block length: about 1.75 * n^(1/3), at least ``min_block``."""
    return int(min(max(round(1.75 * n ** (1.0 / 3.0)), min_block, 1), n))


def block_bootstrap_indices(n: int, block: int, rng: np.random.Generator) -> NDArray[np.intp]:
    """Draw indices for one moving-block bootstrap sample of length ``n``.

    Blocks of ``block`` consecutive positions are drawn with replacement (starting points
    uniform over all n - block + 1 positions) and concatenated, so short-range
    autocorrelation inside each block is preserved.
    """
    n_blocks = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
    return idx.astype(np.intp)
