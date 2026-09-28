# alphafade progress log

Resume here. The original brief is in `SPEC.md`. This file records what's done, what's next,
and every decision that refines or overrides the spec, with its reason.

## Status

| Milestone | State |
|---|---|
| M0 Plan, open questions, layout, API | Done (approved 2026-09-26) |
| M1 Validation/alignment + ic_series, rolling_ic, rolling_sharpe | Done 2026-09-26: 56 passed, 99% coverage |
| M2 fit_decay (bootstrap CI, linear fallback) | Done 2026-09-27: 80 passed, decay.py 99% coverage |
| M3 find_break, chow_test, publication_gap (Newey-West) | Not started |
| M4 crowding_score | Not started |
| M5 analyze/FadeReport, plotting, Ken French loader, README, UMD example | Not started |
| M6 CI, packaging, TestPyPI release | Not started |

**Next action:** see the first milestone in the table that isn't Done. Jeevun asked (2026-09-26)
for M1–M6 to be built straight through without stopping for approval between milestones.

---

## M0 plan (proposed 2026-09-26)

### Goal restated
alphafade tells a researcher whether a trading signal's edge is shrinking over calendar time
(months and years), how fast, and whether that coincides with crowding. The user brings the
data: a strategy return series, or a signal panel plus forward returns, and optionally
stock-level returns with long/short leg membership. alphafade returns rolling quality measures,
a fitted decay curve with an honest half-life and confidence interval, structural-break and
publication-gap tests with autocorrelation-robust statistics, and a comomentum crowding score.
Every statistic has to stay honest under overlapping windows, look-ahead risk, and noisy data,
and "no detectable decay" is a valid, reported answer.

### Research findings that shaped the plan (verified 2026-09-26)
- **Lou & Polk comomentum is leave-one-out, not pairwise.** For each stock in a decile, they
  take the correlation between its FF3 residual and the equal-weighted residual of the *rest*
  of the decile, then average across stocks. They use weekly returns over the 12-month ranking
  period and report losers and winners separately (CoMOM_L, CoMOM_W). The spec says "average
  pairwise", which is a related but different number. See open question Q1.
- **McLean & Pontiff:** confirmed about 26% lower out-of-sample and about 58% lower after
  publication. Their method is a regression of returns on a post-sample dummy and a
  post-publication dummy, with each coefficient divided by the in-sample mean. publication_gap
  will use the same regression.
- **statsmodels has no sup-F (Andrews) test.** It has OLS-CUSUM (`breaks_cusumolsresid`) and
  Hansen (1992). Published critical values for one parameter at 15% trimming disagree slightly
  between Andrews (1993), where 5% = 8.85, and later textbook tables, where 5% = 8.68. So we
  simulate the limiting distribution ourselves with a fixed seed, ship the resulting table, and
  test it against both published sets.
- **Ken French files:** CSVs inside zips, values in **percent**, `YYYYMM` or `YYYYMMDD` dates,
  and the monthly and annual tables are stacked in one file. The parser must stop at the first
  blank line. Relevant files: `F-F_Research_Data_Factors[_weekly|_daily]_CSV.zip` and
  `F-F_Momentum_Factor[_daily]_CSV.zip`.
- **Nothing else does this.** alphalens(-reloaded), pyfolio(-reloaded), and quantstats don't
  measure calendar-time decay, half-lives, breaks, or crowding. ruptures does generic
  change-point detection with no finance statistics. A GitHub repo `tugsuz/alpha-decay` does
  state-space half-lives but has 0 stars and isn't on PyPI.
- **Python and library support:** Python 3.10 reaches end-of-life on 2026-10-31. pandas 3.x
  requires Python 3.11+, and numpy 2.5 and scipy 1.18 require 3.12+. Supporting 3.10 means
  users on 3.10 get pandas 2.x, numpy ≤2.2, and scipy ≤1.15, so the code must work on
  pandas 2.2 and 3.x either way. See Q2.

### Proposed changes to the API sketch (and why)
1. **Add `forward_returns(returns, periods=1)`.** This is the single, tested place where
   realized returns get shifted so that row t holds the return earned *after* t. Look-ahead
   bugs mostly come from users doing this shift by hand.
2. **Add `ic_series(signal, fwd_returns)`.** It gives the per-date IC, and `rolling_ic` is just
   its rolling mean. This matters because `fit_decay` should be fed the *raw* per-period IC
   or returns, not a rolling average. Rolling windows overlap, which fakes smoothness,
   inflates R², and shrinks confidence intervals. Rolling series are still accepted, but
   they carry their window length in `.attrs`, and `fit_decay` then widens its bootstrap
   blocks and warns.
3. **Rename `test_break_at` to `chow_test`.** pytest collects any imported function whose name
   starts with `test_`, so `from alphafade import test_break_at` inside a user's test file
   would be run as a test.
4. **Parameterize decay by rate λ (per year), not τ.** "No decay" is λ = 0, where τ would be
   infinite, and a bootstrap handles λ cleanly. The half-life is ln(2)/λ, reported only when
   the confidence interval for λ lies entirely above 0. Otherwise the result says "no
   detectable decay" and `half_life_years` is `None`.
5. **Use a residual block bootstrap for the decay CI.** It keeps the time axis fixed and
   resamples blocks of residuals, which preserves autocorrelation. A plain resample of
   (t, y) pairs would scramble the trend being measured.
6. **Tentative: drop statsmodels from runtime dependencies.** Everything we need (Newey-West
   OLS, CUSUM, simple regressions) is about 60 lines of numpy. statsmodels would stay as a
   *test oracle*: our Newey-West numbers must match it to 1e-10. That gives a smaller install,
   faster imports, and no untyped dependency fighting `mypy --strict`. See Q3.
7. **Tentative: use stdlib `urllib` instead of `requests`** for Ken French downloads. That
   removes the `[data]` extra entirely. See Q3.
8. **Tentative: one small addition to `analyze`, a "crowding link".** If a crowding score is
   supplied, run a Newey-West regression of next-12-month strategy return on the crowding
   level. That's the "does fading line up with crowding" number the mission asks for, and
   Lou & Polk's main result is exactly this kind of relationship. See Q4.

### Package layout
```
pyproject.toml            hatchling, PEP 639 license = "MIT", uv dependency groups
README.md  CHANGELOG.md  LICENSE  CONTRIBUTING.md  PROGRESS.md  SPEC.md  CLAUDE.md
src/alphafade/
  __init__.py             public exports + __version__ (the ONLY public surface)
  py.typed                ships type hints (PEP 561)
  _errors.py              exception + warning classes
  _validate.py            input coercion, alignment, frequency inference, NaN reporting
  _stats.py               Newey-West OLS, lag rule, moving-block bootstrap, sup-Wald p-values
  _supwald_table.py       generated quantile table (see scripts/)
  rolling.py              forward_returns, ic_series, rolling_ic, rolling_sharpe
  decay.py                fit_decay, DecayFit
  breaks.py               find_break, chow_test, BreakResult
  publication.py          publication_gap, GapResult
  crowding.py             crowding_score
  report.py               analyze, FadeReport
  plotting.py             lazily imported; clear ImportError telling you to install [plot]
  datasets.py             load_ff3, load_momentum (explicit download, cached)
scripts/make_supwald_table.py   reproducible, seeded simulation of the Andrews distribution
tests/                    one test file per module + test_alignment.py + test_properties.py
examples/umd_momentum.py  end-to-end on the real UMD factor
docs/methodology.md       the math, assumptions, and citations
.github/workflows/ci.yml, release.yml
```

### Public API (everything else is private)
```python
import alphafade as af

# M1: alignment and rolling quality
fwd  = af.forward_returns(returns, periods=1)                   # row t = return over (t, t+1]
ic_t = af.ic_series(signal, fwd, method="spearman", min_assets=5)   # per-date IC (Series)
ic   = af.rolling_ic(signal, fwd, window=36, method="spearman", min_assets=5)
sr   = af.rolling_sharpe(returns, window=252, freq=None, rf=0.0)    # annualized

# M2: decay
fit = af.fit_decay(ic_t, n_boot=1000, alpha=0.05, block_size=None, rng=42)
#  DecayFit(model, decay_detected, half_life_years, ci_low, ci_high, decay_rate_per_year,
#           rate_ci, a, r2, linear_slope_per_year, linear_t_stat, better_fit, aic_exp,
#           aic_linear, n_obs, fitted, notes)

# M3: breaks and publication
brk = af.find_break(returns, trim=0.15, hac_lags=None)    # sup-Wald, HAC; method="cusum" opt.
chw = af.chow_test(returns, date="2000-01-01", hac_lags=None)
#  BreakResult(date, stat, p_value, mean_before, mean_after, method, n_obs)
gap = af.publication_gap(returns, sample_end="1989-12-31", publication_date="1993-03-01")
#  GapResult(table: DataFrame[in_sample/post_sample/post_publication x
#            start,end,n,mean,ann_mean,ann_sharpe,t_stat,decline_pct],
#            post_sample_change, post_sample_t, post_pub_change, post_pub_t, hac_lags)

# M4: crowding
crowd = af.crowding_score(stock_returns, long_members, short_members, factors=ff3_weekly,
                          window=52, method="leave_one_out", min_obs=26)
#  DataFrame indexed by formation date: long, short, mean, n_long, n_short

# M5: everything together
report = af.analyze(returns, signal=signal, fwd_returns=fwd, sample_end="1989-12-31",
                    publication_date="1993-03-01", crowding=crowd, rng=42)
print(report.summary()); report.to_frame(); report.plot()

ff3 = af.datasets.load_ff3(freq="weekly")      # cached in ~/.cache/alphafade/, decimals
umd = af.datasets.load_momentum(freq="monthly")

# Errors and warnings (all exported)
af.AlphaFadeError > af.InputError(ValueError) > af.AlignmentError / af.FrequencyError /
                                                af.InsufficientDataError
af.AlphaFadeWarning(UserWarning) > af.DataDroppedWarning / af.FitWarning
```

### Key conventions (these apply to every milestone)
- **Inputs:** pandas objects with a DatetimeIndex. Panels are wide: dates × assets. Long-format
  or unsorted data raises an `InputError` that says how to reshape it.
- **Alignment:** `signal` and `fwd_returns` are aligned by exact date and asset labels. No
  silent reindexing: if labels don't overlap, raise; if some don't, warn with counts.
- **Frequency:** inferred from median date spacing into D/W/M/Q/A (252/52/12/4/1 periods per
  year). Mixed or irregular spacing raises `FrequencyError` with a hint to pass `freq=` or
  resample. Two inputs with different frequencies raise; we never resample for you.
- **Time axis for decay:** actual elapsed years (days / 365.25), not row counts, so gaps are
  handled correctly.
- **Newey-West lags:** default floor(4·(T/100)^(2/9)). For series built from overlapping
  windows, at least window − 1. `hac_lags=` overrides.
- **Randomness:** `rng: int | np.random.Generator | None`. Docs say to pass a seed.
- **Lossy steps** (dropping NaNs, dates with too few assets, stocks with too few obs) always
  emit a `DataDroppedWarning` with counts.
- **Docstrings:** NumPy style with a runnable `Examples` section. Doctests run in CI.
- **Survivorship warning** goes in the docstrings for `ic_series` and `crowding_score` and in
  the README Limitations section.

### Tooling
uv (venv + lock), hatchling, ruff (lint + format), mypy --strict on `src/` (pandas-stubs,
scipy-stubs), pytest + pytest-cov + hypothesis, and statsmodels in the dev group as an oracle.
Coverage gate: ≥90% on rolling/decay/breaks/publication/crowding. CI: GitHub Actions matrix of
Python versions × {ubuntu, macos}, plus a job pinned to the lowest supported pandas/numpy/scipy
(`uv sync --resolution lowest-direct`), so the minimum versions we claim are actually tested.
Release: tag `v*` → build → TestPyPI → PyPI, trusted publishing, `environment: pypi` /
`testpypi` (same pattern as finlearn-analytics).

### Validation tests (from spec, plus additions)
Spec: exponential IC recovery within 10%, constant IC gives no decay, planted break found,
crowding rises with factor loading and stays near 0 when independent, hand-computed 3×4 IC.
Additions: look-ahead test (a signal equal to the *current* return must give IC ≈ 0 against
forward returns, while a signal equal to the forward return gives IC = 1); Newey-West matches
statsmodels; the sup-Wald table matches published critical values; publication_gap recovers
planted 26%/58% declines; Ken French parser works on a checked-in sample file (no network in
tests); hypothesis properties (Spearman IC unchanged by monotone transforms, IC in [-1, 1],
Sharpe unchanged by scaling).

### Open questions (need Jeevun's answers)
- **Q1 Comomentum definition.** Default to Lou & Polk's leave-one-out version (faithful to
  the paper) with `method="pairwise"` available as in the spec? *Recommend: yes.*
- **Q2 Python versions.** Keep ≥3.10 (spec), or move to ≥3.11 since 3.10 hits end-of-life
  about 5 weeks after we'd ship? Also add 3.14 to CI? *Recommend: ≥3.11, CI 3.11–3.14.*
- **Q3 Dependencies.** Drop statsmodels and requests from runtime, keeping statsmodels as a
  test-only oracle? *Recommend: yes.* Core becomes numpy, pandas, and scipy.
- **Q4 Crowding link in `analyze`.** Add the Newey-West regression of future strategy return on
  crowding level? It's a small addition, but it's beyond the literal spec. *Recommend: yes.*
- **Q5 UMD example dates.** UMD is momentum, so the paper is Jegadeesh & Titman (1993): sample
  1965–1989, published March 1993. The sketch's dates (1993/1997) look like Carhart (1997).
  Use Jegadeesh & Titman? *Recommend: yes.*
- **Q6 Repo location and hosting.** Build right here (`~/Coding/Python. #2`) and publish as a
  public GitHub repo `JSand15/alphafade`? And do you already have PyPI and TestPyPI accounts
  (needed for trusted publishing in M6)?
- **Q7 Docs.** For v0.1, README + docstrings + `docs/methodology.md`, with no hosted docs site?
  *Recommend: yes, and add a hosted site later if people use it.*

## Decisions log
| Date | Decision | Reason |
|---|---|---|
| 2026-09-26 | Spec wins over finlearn conventions where they conflict (hatchling vs setuptools, NumPy vs Google docstrings) | SPEC.md says so explicitly |
| 2026-09-26 | Borrow from finlearn: PEP 639 `license = "MIT"`, Keep a Changelog, CONTRIBUTING.md, trusted-publishing release workflow with `environment: pypi` | Consistency across Jeevun's libraries |
| 2026-09-26 | Custom exceptions subclass `ValueError` | Specific errors per spec, but `except ValueError` still works for users |
| 2026-09-26 | Name `alphafade` is free on PyPI and TestPyPI (checked 2026-09-26) | Needed before building |
| 2026-09-26 | **M0 approved.** Jeevun accepted all 7 recommendations and all API changes, and asked for M1–M6 to be built straight through | Jeevun's reply |
| 2026-09-26 | Q1: comomentum defaults to leave-one-out, with `method="pairwise"` also available | Faithful to Lou & Polk |
| 2026-09-26 | Q2: `requires-python >= 3.11`, CI on 3.11–3.14 | 3.10 reaches end-of-life 2026-10-31 |
| 2026-09-26 | Q3: runtime deps are numpy, pandas, scipy only. statsmodels is a dev-only oracle. Downloads use stdlib urllib, so there's no `[data]` extra | Smaller install, clean mypy --strict |
| 2026-09-26 | Q4: `analyze` reports a crowding link (Newey-West regression of forward strategy return on crowding) | Answers the mission's "why" |
| 2026-09-26 | Q5: UMD example uses Jegadeesh & Titman (1993): sample end 1989-12-31, published 1993-03-01 | UMD is momentum |
| 2026-09-26 | Q6: public GitHub repo JSand15/alphafade, built in this folder | Jeevun approved |
| 2026-09-26 | Q7: v0.1 docs are README + docstrings + docs/methodology.md, with no hosted docs site | Keep scope tight |
| 2026-09-26 | Frequency bands (days between dates): D 1–5, W 5.5–10, M 25–35, Q 80–100, A 350–380. Out-of-band gaps may cover at most 5% of the time span | Count-based rules would let 5 years of monthly data hide inside 20 years of daily data; weighting by time catches it |
| 2026-09-26 | Newey-West has no n/(n−k) small-sample correction | Matches statsmodels' default and the original Newey & West (1987) paper; verified to 1e-10 |
| 2026-09-26 | Leading and trailing NaNs are trimmed silently; NaNs in the middle warn | Edge NaNs come from rolling warm-up and forward shifts and carry no information |
| 2026-09-26 | `forward_returns` compounds by multiplying shifted (1 + r) terms, not by summing log returns | Summing logs turns a −100% return into −inf and then NaN in pandas' rolling sum |
| 2026-09-26 | A constant window (max == min) gets NaN rolling Sharpe | pandas' streaming variance leaves float residue on constant windows, which would give a huge fake Sharpe |
| 2026-09-27 | Linear-fallback decay is only claimed if the fitted starting level has a Newey-West \|t\| ≥ 2 | The linear rate is relative to the starting level; a line rising from ~0 would otherwise be misread as "a negative edge shrinking" |
| 2026-09-27 | Decay p-value = share of bootstrap rates ≤ 0 (one-sided); detection = whole (1−alpha) CI for the rate above 0 | Simple and honest; "no detectable decay" whenever the CI includes 0 |
| 2026-09-26 | Tests fail on any unexpected AlphaFadeWarning (autouse fixture in tests/conftest.py) | Forces every lossy path to be tested deliberately. It's a fixture, not ini filterwarnings, so coverage still measures import-time lines |

## M2 notes
- Exponential fit uses variable projection: for each rate, the best level `a` has a
  closed form, so the fit is a 1-D search (grid, then vectorized golden section) and can't
  diverge. Grid bounds: half-life ≥ max(1% of span, 3 observations), and growth ≤ 100× over
  the sample.
- The spec's "within 10%" test needs a precise sample. With realistic noise (3000 assets,
  60 years monthly) single estimates scatter about 6% (sd), so about 1 in 4–7 draws miss 10%.
  Tests therefore check: (a) within 10% at per-date noise sd 0.005 over 80 years; (b) median
  error < 3% across 25 seeds at realistic noise (unbiasedness); (c) the end-to-end
  asset-panel pipeline within 35%. This precision fact goes in the README Limitations.
- Constant IC false-positive rate is about 2.5% (1/40 in a probe), as designed for a
  two-sided 95% CI.
