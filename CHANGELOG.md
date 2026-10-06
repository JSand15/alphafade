# Changelog

All notable changes to **alphafade** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-06

### Added
- `forward_returns`: the one look-ahead-safe way to line up signals with future returns.
- `ic_series`, `rolling_ic`: per-date and rolling information coefficient (Spearman or Pearson).
- `rolling_sharpe`: annualized rolling Sharpe with frequency inference.
- `fit_decay` / `DecayFit`: exponential decay of an edge over calendar time with half-life,
  residual block-bootstrap confidence interval, linear comparison (AIC), linear fallback, and
  an explicit "no detectable decay" result.
- `find_break`: unknown-date break search (Andrews sup-Wald with a simulated p-value table, or
  CUSUM); `chow_test` for a known date; `BreakResult`.
- `publication_gap` / `GapResult`: McLean & Pontiff in-sample / post-sample / post-publication
  comparison with Newey-West t-statistics.
- `crowding_score`: Lou & Polk comomentum (leave-one-out, or pairwise) per trade leg.
- `analyze` / `FadeReport` / `CrowdingLink`: everything in one call, with `.summary()`,
  `.verdict()`, `.to_frame()`, `.to_dict()`, `.to_json()` and `.plot()` (matplotlib via the
  `plot` extra). With short history the default rolling window shrinks to a third of the data.
- `signal_lifetime` / `LifetimeResult`: years (and calendar date) until a fitted edge falls to
  a chosen level or share of its start, with a confidence interval.
- `compare_signals` / `SignalComparison`: fit and rank many signals by decay speed, with Holm
  or Benjamini-Hochberg multiple-testing adjustment.
- `walk_forward_decay` / `WalkForwardResult`: expanding-window refits with no look-ahead and
  stability measures for the half-life.
- `ic_by_horizon` / `HorizonResult`: mean IC per forecast horizon (Newey-West t) and a horizon
  half-life.
- `datasets.load_ff3`, `datasets.load_momentum`: Ken French factors as decimals, downloaded
  only when asked, cached in `~/.cache/alphafade/` (or `$ALPHAFADE_CACHE`), or offline via `path=`.
- Specific exceptions (`InputError`, `AlignmentError`, `FrequencyError`,
  `InsufficientDataError`, `DownloadError`) and warnings (`DataDroppedWarning`, `FitWarning`).
- Safety: downloads are https-only (redirects included), time-limited, and size-capped; zip
  files are decompressed with a cap on the real output size. Boolean data, overflow-scale
  values, and Newey-West lags below the overlap floor are rejected with clear errors.
- PEP 561 `py.typed` marker; `mypy --strict` clean.
- Example: `examples/umd_momentum.py` (real momentum factor, 1963 to present).

[Unreleased]: https://github.com/JSand15/alphafade/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/JSand15/alphafade/releases/tag/v0.1.0
