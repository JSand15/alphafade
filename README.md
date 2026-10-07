# alphafade

[![CI](https://github.com/JSand15/alphafade/actions/workflows/ci.yml/badge.svg)](https://github.com/JSand15/alphafade/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/alphafade.svg)](https://pypi.org/project/alphafade/)
[![Python](https://img.shields.io/pypi/pyversions/alphafade.svg)](https://pypi.org/project/alphafade/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/JSand15/alphafade/blob/main/LICENSE)

**Is my trading signal dying, and if so, how fast and why?**

Every trading edge fades as more people find it. alphafade tells you whether yours has, how
fast, and whether the fade lines up with crowding (more money chasing the same pattern). It
gives you an honest answer: a half-life with a confidence interval, or "no detectable decay"
when the data can't tell.

Most tools measure how a signal's power fades over the days after each trade. alphafade
measures something different: how the edge shrinks across *calendar time* (months and years).

## What you get

- **A decay half-life.** How many years until the edge halves, with a bootstrap confidence
  interval, compared against a straight-line fade.
- **Break and publication tests.** Did the edge drop at a date you chose, at an unknown date,
  or after the academic paper came out (the McLean & Pontiff 2016 effect)?
- **A crowding score.** Lou & Polk comomentum: how much a trade's stocks move together.
- **Lifetime forecasts.** `signal_lifetime`: when does a fading edge reach a level you care about?
- **Many signals at once.** `compare_signals` ranks them and corrects for testing many (otherwise
  some pure-noise signal always looks like it is fading by luck).
- **A stability check.** `walk_forward_decay` refits as data arrives, so you can see whether the
  half-life is real or a quirk of one sample, with no look-ahead.
- **Forecast-horizon decay.** `ic_by_horizon`: how many periods ahead a signal still predicts.
- **One report.** `analyze()` runs it all and gives plain English, a chart, or JSON.

## Who it's for

Quant researchers and students who want to know whether a published anomaly still works,
analysts monitoring a live strategy for decay, and anyone testing the claim that "alpha
decays". No finance degree needed to read the output: every result comes with a plain-English
summary.

## Why trust it

- Every t-statistic is autocorrelation-robust (Newey-West) and checked against statsmodels.
- Confidence intervals were tested by simulation: they contain the true half-life 89-95% of
  the time, and flag decay in pure noise only 2-6% of the time.
- Nothing is dropped, resampled, or shifted behind your back; anything lossy raises a warning.
- Bad inputs get a clear error that says how to fix them, never a confident wrong number.
- 500+ tests on Python 3.11 to 3.14, strict type checking, and a fully documented method
  (every formula is in [`docs/methodology.md`](https://github.com/JSand15/alphafade/blob/main/docs/methodology.md)).

## Install

```bash
pip install alphafade             # core: numpy, pandas, scipy
pip install "alphafade[plot]"     # + matplotlib for report.plot()
```

Python 3.11 or newer. With [uv](https://docs.astral.sh/uv/): `uv add "alphafade[plot]"`.

## Quickstart

Is momentum dying? This downloads Ken French's momentum factor (a few KB, cached
afterwards) and runs every analysis:

```python
import alphafade as af

umd = af.datasets.load_momentum("M").loc["1963-07":]  # monthly momentum returns
report = af.analyze(
    umd,
    sample_end="1989-12-31",        # Jegadeesh & Titman's sample ended here
    publication_date="1993-03-01",  # ...and their paper came out here
    rng=42,                         # reproducible bootstrap
)
print(report.summary())
report.plot()                       # needs alphafade[plot]
```

Part of the output (data through August 2026):

```text
Verdict: No detectable decay in the strategy's average return so far.
...
  After publication (1993-03 to 2026-08, 402 obs): average 0.003823 per period (4.59% a year,
  Sharpe 0.28, t = 1.60), 53% lower than in-sample (t of the change = -1.40).
```

Momentum earns about half as much since publication, but it's so volatile that 33 years of
data can't rule out luck. alphafade tells you that instead of printing a confident-looking
half-life.

### With a signal instead of returns

If you have the signal itself (dates × assets) and asset returns, the per-date
information coefficient (IC: the cross-sectional correlation between today's signal and
the returns that follow) is usually the sharper thing to track:

```python
import numpy as np, pandas as pd
import alphafade as af

rng = np.random.default_rng(0)
dates = pd.date_range("1980-01-31", periods=480, freq="ME")      # 40 years, 500 stocks
signal = pd.DataFrame(rng.standard_normal((480, 500)), index=dates)
edge = 0.10 * np.exp(-np.arange(480) / 12 / 5)                   # true half-life: 3.5 years
realized = 0.05 * (edge[:, None] * signal + rng.standard_normal((480, 500))).shift(1)

fwd = af.forward_returns(realized)   # row t = return earned AFTER t (no look-ahead)
ic = af.ic_series(signal, fwd)       # Spearman IC per date
fit = af.fit_decay(ic, rng=0)        # exponential decay + bootstrap CI
print(fit.summary())
```

```text
Exponential decay fit on 479 observations (1980-01 to 2019-11).
The edge started at 0.07159 and is shrinking about 14.8% per year: half-life 4.3 years (95% CI 3.0 to 5.9).
The exponential model fits better than the linear one (AIC 22.6 lower).
```

The true half-life (3.5 years) is inside the interval. Across 12 random seeds of this
setup the median estimate was 3.56 years and every interval contained the truth.

### Beyond the basics

Four more tools, each answering a question a half-life alone can't:

- `signal_lifetime` turns a fit into "when does the edge reach a level I care about?"
- `compare_signals` ranks many strategies at once and corrects the p-values for testing many
  signals (otherwise some pure-noise signal will look like it is fading by luck).
- `walk_forward_decay` refits at each date using only the data available then, so you can see
  whether the half-life is stable or just a quirk of the full sample.
- `ic_by_horizon` measures fading across the *forecast horizon* (how many periods ahead the
  signal still predicts), which is a different question from fading across calendar time.

```python
import numpy as np, pandas as pd
import alphafade as af

rng = np.random.default_rng(1)
dates = pd.date_range("1980-01-31", periods=480, freq="ME")
years = np.arange(480) / 12
fading = pd.Series(0.02 * np.exp(-years / 8) + rng.normal(0, 0.02, 480), index=dates)
noise = pd.Series(rng.normal(0.003, 0.02, 480), index=dates)

fit = af.fit_decay(fading, rng=0)
life = af.signal_lifetime(fit, fraction=0.5)     # when is the edge down to half its start?
print(life.summary())

ranked = af.compare_signals(pd.DataFrame({"fading": fading, "noise": noise}), rng=0)
print(ranked.table[["half_life_years", "p_adjusted", "decay_detected_adjusted"]].round(3))

walk = af.walk_forward_decay(fading, min_obs=120, step=60, rng=0)
print(walk.table[["n_obs", "decay_detected"]].tail(3))
```

```text
The fitted exponential edge reaches 50% of its starting level about 3.6 years after the sample start (95% CI 2.3 to 5.7), around 1983-08. That has already happened. The interval holds the starting level fixed and only varies the decay rate.
        half_life_years  p_adjusted  decay_detected_adjusted
fading            3.554       0.000                     True
noise               NaN       0.129                    False
            n_obs  decay_detected
2009-12-31    360            True
2014-12-31    420            True
2019-12-31    480            True
```

The true half-life here is 5.5 years and the interval contains it. Multiple-testing correction
lowers, but cannot remove, the chance that pure noise is flagged: with other seeds this same
setup occasionally flags the noise series too. `FadeReport.to_dict()` and `.to_json()` export
a report's headline numbers as plain data.

## Public API

| Function / class | What it does |
|---|---|
| `forward_returns(returns, periods=1)` | Shift realized returns so row *t* holds the return earned after *t*. The one place alphafade moves data in time. |
| `ic_series(signal, fwd_returns, method="spearman")` | Per-date information coefficient. |
| `rolling_ic(signal, fwd_returns, window=36)` | Rolling mean IC. |
| `rolling_sharpe(returns, window=252)` | Annualized rolling Sharpe ratio (frequency inferred). |
| `fit_decay(perf, n_boot=1000, rng=None)` → `DecayFit` | Exponential decay *a*·e^(−λt) over calendar years: half-life, block-bootstrap CI, linear comparison (AIC), linear fallback. |
| `find_break(returns, method="sup_wald")` → `BreakResult` | Unknown-date break search (Andrews sup-Wald, or CUSUM). |
| `chow_test(returns, date)` → `BreakResult` | Did the average change at a date you chose in advance? |
| `publication_gap(returns, sample_end, publication_date)` → `GapResult` | McLean & Pontiff split: in-sample / post-sample / post-publication means, Sharpe, % declines, Newey-West t-stats. |
| `crowding_score(stock_returns, long_members, short_members, factors=...)` | Lou & Polk comomentum per leg (leave-one-out or pairwise residual correlation). |
| `signal_lifetime(fit, floor=None, fraction=None)` → `LifetimeResult` | Years until a fitted edge falls to a level (or share of its start), with a CI and a calendar date. |
| `compare_signals(perf)` → `SignalComparison` | Fit and rank many signals by decay speed, with Holm or Benjamini-Hochberg adjusted p-values. |
| `walk_forward_decay(perf, min_obs=60, step=12)` → `WalkForwardResult` | Expanding-window refits with no look-ahead; shows whether the half-life is stable. |
| `ic_by_horizon(signal, returns, horizons)` → `HorizonResult` | Mean IC (Newey-West t) per forecast horizon and the horizon half-life. |
| `analyze(returns, ...)` → `FadeReport` | Everything above, with `.summary()`, `.verdict()`, `.to_frame()`, `.to_dict()`, `.to_json()`, `.plot()`. |
| `datasets.load_ff3(freq)`, `datasets.load_momentum(freq)` | Ken French factors as decimals: explicit download, cached in `~/.cache/alphafade/`, or offline with `path=`. |

Errors are specific and say how to fix the input: `InputError` (a `ValueError`),
`AlignmentError`, `FrequencyError`, `InsufficientDataError`, `DownloadError`, all subclasses
of `AlphaFadeError`; warnings subclass `AlphaFadeWarning`. Anything lossy
(dropping NaNs, thin cross-sections) emits a `DataDroppedWarning` with counts; unreliable
fits emit `FitWarning`. Every function that uses randomness takes `rng=` (a seed or a numpy
`Generator`).

## How this differs from alphalens and quantstats

| | alphalens | quantstats / pyfolio | **alphafade** |
|---|---|---|---|
| Question | How good is this factor, and over what forecast horizon does its IC fade? | How did this portfolio perform (returns, drawdowns, risk)? | Is the edge shrinking across *years*, how fast, since when, and is crowding to blame? |
| Time axis | Days after the signal (forecast horizon) | Calendar time, descriptive | Calendar time, *inferential* |
| Half-life with confidence interval | No | No | Yes (block bootstrap; "no detectable decay" when appropriate) |
| Structural breaks, publication effect | No | No | sup-Wald, CUSUM, Chow, McLean & Pontiff regression |
| Crowding | No | No | Lou & Polk comomentum |
| Autocorrelation-robust t-stats | No | No | Newey-West everywhere |

They're complements: use alphalens to build and vet a factor, quantstats to report on a
portfolio, and alphafade to ask whether the edge is going away. alphafade isn't a
backtester and doesn't build portfolios; you bring returns or a signal.

## Limitations (read these)

- **Decay is hard to measure.** With a realistic signal (3,000 stocks, 60 years of monthly
  data, starting IC 0.10), single half-life estimates scatter by about ±6% (one standard
  deviation). For a volatile strategy's raw returns, decades of data often can't
  distinguish decay from noise, as the UMD example shows. Wide intervals are the honest
  answer, not a bug.
- **One shift, not many.** `find_break` looks for a single change in the average. Several
  regime changes, or a slow slide, show up as one "most likely" date. Use `fit_decay` to
  describe a gradual fade.
- **Asymptotic p-values.** sup-Wald p-values come from a simulated large-sample
  distribution (reproducible: `scripts/make_supwald_table.py`) and are floored at 0.0005.
  In simulations its false-alarm rate is close to 5% at a few hundred observations but can
  rise under strong autocorrelation. It gives up a little power to stay honest.
- **Decay toward zero.** The exponential model assumes the edge fades to 0, not to some
  permanent floor. A linear trend is always fitted alongside for comparison.
- **Survivorship bias.** If your stock universe contains only companies that exist today,
  the IC and crowding scores are biased (the losers that got delisted are missing). Use a
  point-in-time universe such as CRSP when you can.
- **Frequencies are never guessed.** Mixed daily/monthly data raises a `FrequencyError`;
  resample it yourself. Inputs must be wide (dates × assets) with a sorted `DatetimeIndex`.
- **Weekly momentum.** Ken French doesn't publish a weekly momentum file; compound the daily
  one.

## Learn more

- [`docs/methodology.md`](https://github.com/JSand15/alphafade/blob/main/docs/methodology.md): every formula and default, with references.
- [`examples/umd_momentum.py`](https://github.com/JSand15/alphafade/blob/main/examples/umd_momentum.py): the end-to-end momentum study above.
- [`CHANGELOG.md`](https://github.com/JSand15/alphafade/blob/main/CHANGELOG.md) · [`CONTRIBUTING.md`](https://github.com/JSand15/alphafade/blob/main/CONTRIBUTING.md)

## References

McLean, R. D., & Pontiff, J. (2016). Does academic research destroy stock return
predictability? *Journal of Finance*, 71(1), 5–32. ·
Lou, D., & Polk, C. (2022). Comomentum: Inferring arbitrage activity from return
correlations. *Review of Financial Studies*, 35(7), 3272–3302. ·
Andrews, D. W. K. (1993). Tests for parameter instability and structural change with
unknown change point. *Econometrica*, 61(4), 821–856. ·
Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and
autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708. ·
Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers.
*Journal of Finance*, 48(1), 65–91.

## License

MIT © 2026 Jeevun Sandhu
