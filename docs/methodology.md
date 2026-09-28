# alphafade methodology

What alphafade computes, why, and where the methods have limits. Every default quoted here is
the default in the code, and each technical term is defined the first time it appears.

## 1. Overview

Most signal-analysis tools study the **forecast horizon**: how a signal's predictive power
fades over the days after each trade (does today's signal still predict returns 5 or 20 days
out?). alphafade studies **calendar time** instead: how the edge of the same signal, used the
same way, shrinks across months and years (did it work better in 1975 than in 2015?). It fits
a decay curve and a half-life, tests for a break at a known or unknown date, compares returns
before and after academic publication (McLean & Pontiff 2016), and measures crowding with a
comomentum score (Lou & Polk 2022). The `analyze()` report bundles these.

## 2. Conventions

### 2.1 Data layout

- Every input has a sorted, duplicate-free, time-zone-free `DatetimeIndex`; anything else
  raises an `InputError` that says how to fix it. Infinite values are rejected.
- A **panel** (signals, stock returns) is a wide DataFrame: one row per date, one column per
  asset. Long format (a MultiIndex) is rejected with a hint to call `.unstack()`.
- Returns are **simple returns as decimals** (0.05 means +5%), not log returns or percent.
- When two panels are combined (`ic_series`), only shared dates and assets are kept. No
  overlap raises an `AlignmentError`; partial overlap emits a `DataDroppedWarning` with counts.

### 2.2 Forward returns and look-ahead

**Look-ahead bias** is accidentally using information that was not yet available on the date
of the decision. `forward_returns(returns, periods=1)` is the one place alphafade moves returns
in time. If `returns` on date $t$ is the return over the period ending at $t$, the output on
date $t$ is the compounded return over the next $p$ = `periods` rows,
$R^{fwd}_t = \prod_{s=1}^{p}(1 + r_{t+s}) - 1$, so a signal known at $t$ is compared with
returns earned strictly after $t$ on the same row. The last $p$ rows are NaN because their
future is not observed. Compounding multiplies $(1 + r)$ terms rather than summing log
returns, because a $-100\%$ return would become $-\infty$ in log space. Values below $-1$
raise an error (they usually mean the input was in percent or log form).

### 2.3 Frequency inference

A function that needs the data frequency (to annualize, or to check that two inputs match)
either takes `freq=` from the user or infers it:

1. Compute the gap in calendar days between consecutive dates, and take the median gap.
2. Pick the frequency whose band contains the median:

   | Code | Meaning | Allowed gap (days) | Periods per year |
   |---|---|---|---|
   | `D` | daily | 1 to 5 | 252 |
   | `W` | weekly | 5.5 to 10 | 52 |
   | `M` | monthly | 25 to 35 | 12 |
   | `Q` | quarterly | 80 to 100 | 4 |
   | `A` | annual | 350 to 380 | 1 |

3. **5% time-share rule.** Add up the days covered by gaps that fall outside that band. If
   they cover more than 5% of the total time span, raise a `FrequencyError` ("mixed or
   irregular frequency") instead of guessing.

Weighting gaps by time rather than counting them matters: five years of monthly data inside
twenty years of daily data is only a few dozen gaps, but it covers a quarter of the span.
Holiday gaps and short market closures cover a tiny share, so they are tolerated. Fewer than
3 dates, or a median gap that fits no band, also raise. Two inputs with different inferred
frequencies (stock returns vs factors in `crowding_score`) raise; alphafade never resamples.
`fit_decay`, `find_break`, and `chow_test` need no frequency: decay uses real elapsed time,
and break tests use the order of the observations.

### 2.4 Missing values (NaN)

`fit_decay`, `find_break`, `chow_test`, and `publication_gap` clean their input the same way.
**Edge** NaNs (leading and trailing) are trimmed silently, because they come from rolling
warm-up and forward-return shifts and carry no information. **Interior** NaNs are real
missing data: they are dropped and a `DataDroppedWarning` reports how many. In `fit_decay`
the time axis comes from the actual dates, so a dropped point leaves a real gap in time.

## 3. Information coefficient (IC)

The **information coefficient** is the cross-sectional correlation, on one date, between the
signal values of all assets and their forward returns. `ic_series(signal, fwd_returns,
method="spearman", min_assets=5)` computes one IC per date from the (date, asset) cells where
both the signal and the forward return are present. **Spearman** (default) ranks the values
within each date (ties get the average rank) and correlates the ranks, which ignores the size
of outliers; **Pearson** correlates the raw values. The IC is NaN on a date with fewer than
`min_assets` usable assets (default 5, minimum 2) or a constant signal or return
cross-section; dates that had data but got NaN are counted in a `DataDroppedWarning`.

`rolling_ic(..., window=36)` is the rolling mean of this series (by default a window needs
all `window` ICs present), and it validates the frequency of the dates.

**Why decay is fitted on the raw per-date IC, not the rolling mean.** Neighbouring values of
a 36-period rolling mean share 35 of their 36 inputs, so the rolling series looks smooth and
trending even when the underlying ICs are independent noise. That is strong
**autocorrelation** (correlation of a series with its own past) created by the method, not
the market, and a regression on it overstates $R^2$ and understates uncertainty. The raw
per-date IC has no built-in overlap. Rolling series are still accepted: they carry their
window length in `.attrs`, and later steps adjust for it (Sections 5.5 and 6).

## 4. Rolling Sharpe ratio

The **Sharpe ratio** is average excess return divided by its standard deviation.
`rolling_sharpe(returns, window=252, freq=None, rf=0.0)` computes, for each window,
$SR_t = \text{mean}(r - r_f) / \text{sd}(r - r_f) \cdot \sqrt{P}$, where the standard
deviation divides by $n-1$ and $P$ is the periods per year (Section 2.3). `rf` is a
per-period risk-free rate, a number or a Series covering every return date; leave it at 0
for long-short (zero-cost) returns. A window of identical returns gets NaN, because
floating-point residue in the rolling variance would otherwise give a huge fake Sharpe.

## 5. Decay fitting (`fit_decay`)

### 5.1 The model and the half-life

`fit_decay(perf, n_boot=1000, alpha=0.05, block_size=None, rng=None)` fits
$p(t) = a\,e^{-\lambda t}$, where $t$ is **years since the first date** (days elapsed / 365.25),
$a$ is the starting level of the edge, and $\lambda$ is the **decay rate** per year
($\lambda > 0$ shrinking, $\lambda = 0$ no decay, $\lambda < 0$ growth). Using a rate rather
than a time constant keeps "no decay" at a finite value. The **half-life**, the time for the
edge to fall to half its level, is $h = \ln 2 / \lambda$. At least 20 non-missing
observations are required, `n_boot` must be at least 100, and $0 < \alpha < 0.5$.

### 5.2 Variable-projection fit

Nonlinear least squares can diverge on noisy data, so alphafade uses **variable
projection**: for any fixed $\lambda$, the best $a$ has a closed form (an ordinary
least-squares projection),

$$a(\lambda) = \frac{\sum_i y_i e^{-\lambda t_i}}{\sum_i e^{-2\lambda t_i}}, \qquad
SSE(\lambda) = \sum_i y_i^2 - a(\lambda)\sum_i y_i e^{-\lambda t_i}.$$

That turns the fit into a one-dimensional search over $\lambda$:

1. **Grid.** Evaluate $SSE$ on 221 candidate rates: 160 log-spaced positive rates from
   $10^{-5}\lambda_{max}$ to $\lambda_{max}$, 60 log-spaced negative rates from
   $10^{-5}\lambda_{min}$ to $\lambda_{min}$, and exactly 0.
2. **Grid bounds.** The shortest half-life allowed is the larger of 1% of the sample span and
   3 times the median spacing between observations, so
   $\lambda_{max} = \ln 2 / \max(\text{span}/100,\ 3\,\Delta t_{median})$. The fastest growth
   allowed is 100 times over the whole sample, so $\lambda_{min} = -\ln(100)/\text{span}$.
3. **Golden-section refinement.** Take the bracket between the grid neighbours of the best
   grid point and run 30 iterations of golden-section search (a bracketing method that
   shrinks the interval by a fixed ratio each step). If the refined point is worse than the
   grid point, the grid point is kept.

### 5.3 Linear comparison and AIC

alphafade also fits a line $p(t) = b_0 + b_1 t$ by ordinary least squares and compares the
two with the **Akaike information criterion** (AIC), a fit score that penalizes parameters
(lower is better): $AIC = n \ln(SSE/n) + 2k$ with $k = 2$ for both models, so it is a fair
comparison. `better_fit` is `"exponential"` when its AIC is lower or equal. $R^2$ is reported
for both; on raw per-period data it is naturally small because single periods are noisy.

### 5.4 Residual moving-block bootstrap

A **bootstrap** estimates uncertainty by re-creating many fake datasets from the real one and
refitting each. alphafade uses a **residual moving-block bootstrap** (Künsch 1989):

1. Take the headline model's residuals and subtract their mean.
2. Build a fake series as fitted values plus residuals drawn in **blocks** of consecutive
   positions (starts drawn uniformly with replacement, blocks joined and cut to length $n$).
   The time axis stays fixed, so the trend is not scrambled, and short-range
   autocorrelation inside each block is preserved.
3. Refit the model to each of `n_boot` fake series (default 1000) and keep the rate.
4. The rate's interval is the **percentile interval** (the $\alpha/2$ and $1-\alpha/2$
   quantiles of the bootstrap rates); the p-value is the share of rates $\le 0$ (one-sided).
   The half-life interval is $[\ln 2 / \lambda_{hi},\ \ln 2 / \lambda_{lo}]$.

Pass `rng` (an int seed or a numpy Generator) for reproducible intervals.

### 5.5 Block length

The default is $\text{round}(1.75\, n^{1/3})$, between 1 and $n$ (16 for 720 monthly
observations). **Rolling inputs:** if `perf` came from `rolling_ic` or `rolling_sharpe`, the
default block is at least the rolling window, and `DecayFit.notes` says $R^2$ is inflated.
An explicit `block_size` is used as given. If $n$ / block < 10, a note says the interval is
rough.

### 5.6 Detection rule and "no detectable decay"

Decay is **detected** only when the whole $(1-\alpha)$ interval for the rate lies above zero
and the point estimate is positive. Otherwise `half_life_years`, `ci_low`, and `ci_high` are
`None` and the summary says "no detectable decay", because a huge half-life from an
insignificant fit would be meaningless. The rule uses one side of a two-sided 95% interval,
so a truly constant IC is flagged about 2.5% of the time (1 in 40 in a probe), as designed.

### 5.7 Linear fallback

If the best exponential rate lands on either end of the grid (decay faster, or growth
steeper, than the data can resolve) or $a$ is not finite, the exponential fit has failed.
alphafade emits a `FitWarning`, adds a note, and uses the linear model for the headline:

- The rate is the fall per year as a share of the starting level, $-b_1 / b_0$. Each bootstrap
  draw refits both the intercept and the slope and uses $-b_1^* / b_0^*$, so the interval
  includes the uncertainty in the starting level (a ratio estimator).
- The linear "half-life" is the time until the line reaches half its starting level,
  $0.5 / (-b_1/b_0)$, with the interval built the same way.
- **Starting-level rule:** decay is only claimed if $b_0$ has a Newey-West $|t| \ge 2$
  (Section 6). A rate relative to a level indistinguishable from zero is meaningless (a line
  rising from about 0 would read as "a negative edge shrinking"), so otherwise detection is
  set to `False` with a note.

### 5.8 How precise is a half-life?

Half-lives are noisy even with a lot of data. In the M2 validation study, with realistic
noise (3000 assets, 60 years of monthly data) single estimates scatter with a standard
deviation of about 6%, so roughly 1 in 4 to 7 draws misses the true half-life by more than
10%, although the median error across 25 seeds is under 3% (close to unbiased). Always read
the confidence interval, not just the point estimate.

## 6. Newey-West standard errors

A **standard error** measures how uncertain an estimate is, and a **t-statistic** is an
estimate divided by its standard error. Ordinary formulas assume independent observations,
which financial series and overlapping windows violate. **Newey-West** standard errors stay
valid under autocorrelation and **heteroskedasticity** (noise whose size changes over time).
For a regression $y = X\beta + u$ with $L$ lags,
$\widehat{\text{Var}}(\hat\beta) = (X'X)^{-1}\hat\Omega(X'X)^{-1}$ with

$$\hat\Omega = \sum_t \hat u_t^2 x_t x_t' + \sum_{l=1}^{L} w_l(\hat\Gamma_l + \hat\Gamma_l'),
\quad \hat\Gamma_l = \sum_{t>l} \hat u_t \hat u_{t-l}\, x_t x_{t-l}', \quad
w_l = 1 - \frac{l}{L+1}.$$

- The weights $w_l$ are the **Bartlett kernel**, which falls linearly and keeps the variance
  positive.
- **Default lags:** $L = \lfloor 4 (n/100)^{2/9} \rfloor$, a common rule of thumb based on
  Newey & West (1994); 4 for $n = 240$. `find_break`, `chow_test`, `publication_gap`,
  and the linear starting-level test raise it to at least window $-1$ for a rolling series. `hac_lags=`
  overrides it ($0 \le L < n$; $L = 0$ gives White, heteroskedasticity-only, errors).
- **No small-sample correction:** no $n/(n-k)$ factor, as in Newey & West (1987) and
  statsmodels' default.
- **Verified** against `statsmodels.OLS(...).fit(cov_type="HAC", cov_kwds={"maxlags": L})`:
  coefficients, standard errors, and t-statistics agree to a relative $10^{-10}$ in tests.

The break tests also use the **long-run variance** of a mean-zero series $e$, the variance of
its average after allowing for autocorrelation, with the same Bartlett weights:
$\hat\sigma^2_{LR} = \frac{1}{n}\left[\sum_t e_t^2 + 2\sum_{l=1}^{L} w_l \sum_{t>l} e_t e_{t-l}\right]$.

## 7. Break tests

A **structural break** is a date at which the average of a series shifts. All break tests
need at least 20 observations and a non-constant series. In every result, `date` is the
first date of the new regime.

### 7.1 Chow-style test at a known date (`chow_test`)

Use this when the date comes from outside the data (a publication, a regulation, a fund
launch). With $D_t = 1$ from `date` on (the first observation on or after it), alphafade
regresses $y_t = c + \delta D_t + u_t$ with Newey-West errors and reports the Wald statistic
$W = t_\delta^2$, which is $\chi^2(1)$ (chi-squared with 1 degree of freedom) under "no
change". At least 5 observations are needed on each side. It is "Chow-style" because the
classic Chow test is an F-test that assumes independent errors, while this version tests only
the mean with autocorrelation-robust errors. Its results match the statsmodels dummy
regression to $10^{-10}$. If you chose the date by looking at the data, use `find_break`.

### 7.2 sup-Wald search for an unknown date (`find_break`, default)

**Candidate range.** With trimming share $\pi_0$ (`trim`, default 0.15), a candidate break
puts $k$ observations before it, for
$k = \max(\lceil \pi_0 n \rceil, 5), \dots, \min(\lfloor (1-\pi_0) n \rfloor, n-5)$.
Trimming exists because a mean can't be estimated from a handful of points at either end.

**Statistic.** For each $k$,
$W_k = (\bar y_{after} - \bar y_{before})^2 / \left[\hat\sigma^2_{LR}\,(1/k + 1/(n-k))\right]$,
and the test statistic is $\sup_k W_k$ (Andrews 1993).

**One long-run variance, from the demeaned series.** $\hat\sigma^2_{LR}$ is computed once
from $y - \bar y$ over the whole sample, not re-estimated around each candidate date. The
reason is **size**, a test's false-alarm rate when there is no break, which should equal the
nominal level. In simulations at $n = 240$ with independent data, the per-date version (a
Newey-West sandwich at each candidate) rejected 9.3% of the time at the 5% level, versus
5.0% for the single variance: taking the maximum over many noisily scaled statistics favours
candidates whose variance was underestimated. The cost is some power (Section 11).

**Break date.** The reported date is at $\arg\max_k W_k$. Because the denominator's variance
is the same for every $k$, and splitting at $k$ reduces the sum of squared residuals by
$\frac{k(n-k)}{n}(\bar y_{after} - \bar y_{before})^2$, this is exactly the **least-squares
break date** (the split that best fits two means). `change_t` is the Newey-West t-statistic
of the dummy regression at that date.

**p-value.** Because the date was searched for, $\sup_k W_k$ is not $\chi^2(1)$. Under no
break it converges to $\sup_{\pi \in [\pi_0, 1-\pi_0]} B(\pi)^2 / (\pi(1-\pi))$, where $B$
is a **Brownian bridge** (a random walk in continuous time pinned to zero at both ends).
`scripts/make_supwald_table.py` simulates this distribution (seed 20260927, 100,000
replications, 5,000 grid steps) and stores 194 quantiles per trim, dense in the right tail,
in `src/alphafade/_supwald_table.py`. The p-value is linearly interpolated in that table.
Critical values for trim 0.15:

| Level | alphafade table | Andrews (1993) | Corrected tables (Andrews 2003) |
|---|---|---|---|
| 10% | 7.20 | 7.17 | 7.12 |
| 5% | 8.76 | 8.85 | 8.68 |
| 1% | 12.28 | 12.35 | 12.16 |

The simulated values sit between the two published sets. Tables are shipped for trims
**0.05, 0.10, 0.15, 0.20, 0.25**; any other `trim` raises an error. p-values are clipped to
the table's range: a statistic above the 99.95% quantile gets the **floor** $p = 0.0005$
(reported as "p < 0.0005"), and one below the 1% quantile gets $p = 0.99$.

### 7.3 CUSUM (`find_break(method="cusum")`)

The **CUSUM** test tracks the cumulative sum of deviations from the overall mean
(Ploberger & Krämer 1992). With $e_t = y_t - \bar y$ and $\hat\sigma_{LR}$ the Newey-West
long-run standard deviation of $e$, the path is
$S_j = \sum_{t \le j} e_t / (\hat\sigma_{LR}\sqrt{n})$ and the statistic is $\max_j |S_j|$.
Under no break the path behaves like a Brownian bridge, so the p-value comes from the
Kolmogorov distribution of $\sup|B|$ (`scipy.stats.kstwobign`). The reported date is the
observation right after the largest excursion. With `hac_lags=0` the long-run variance is
the plain variance (divisor $n$), and the statistic and p-value equal statsmodels
`breaks_cusumolsresid(resid, ddof=0)` on the demeaned series (tested to $10^{-10}$ and
$10^{-8}$). Using the long-run standard deviation keeps autocorrelation from faking a drift.

## 8. Publication gap (`publication_gap`)

Following McLean & Pontiff (2016), the series is split into three periods, exactly as coded:

| Period | Dates included |
|---|---|
| in-sample | $\text{date} \le$ `sample_end` |
| post-sample | `sample_end` $<$ date $<$ `publication_date` |
| post-publication | $\text{date} \ge$ `publication_date` |

`publication_date` must be after `sample_end`. If `sample_start` is given, earlier
observations are dropped with a `DataDroppedWarning`. At least 12 in-sample observations and
at least 1 post-publication observation are required; the post-sample period may be empty.

**Regression.** With dummies $D^{ps}_t$ and $D^{pub}_t$ for the later periods, alphafade
fits $r_t = a + b_1 D^{ps}_t + b_2 D^{pub}_t + u_t$ with Newey-West errors (default lag rule
on the full sample, at least window $-1$ for a rolling series, or `hac_lags=`). $a$ equals the in-sample mean, and $b_1$, $b_2$ are how
much higher (positive) or lower (negative) each later period's mean is. If the post-sample
period is empty, its dummy is left out.

**Declines.** The percentage declines are $-b_1/a$ and $-b_2/a$ (0.58 means 58% lower). They
are NaN when $a \le 0$, because a percentage decline of a non-positive mean is meaningless,
and the post-sample decline is also NaN when that period is empty.

**Publication vs post-sample.** `publication_vs_post_sample_t` tests $b_2 - b_1$ with the
contrast vector $c = (0, -1, 1)$: $t = (b_2 - b_1)/\sqrt{c' \hat V c}$, using the full
Newey-West covariance. A large $|t|$ means publication itself mattered beyond ordinary
out-of-sample decay. It is NaN when the post-sample period is empty.

**Per-period table:** start, end, `n_obs`, mean, annualized mean ($\times P$), annualized
Sharpe (mean / sd with $n-1$, $\times\sqrt{P}$, no risk-free rate), and a Newey-West t of the
period mean, which needs more than 2 observations. Its lags are `hac_lags` (capped at the
period length minus 1) when given; otherwise the default rule on that period's own length,
raised to window $-1$ for a rolling series.

## 9. Crowding (`crowding_score`)

**Comomentum** is the average correlation of the **residual returns** (returns left over
after removing what a factor model explains) among stocks in the same leg of a trade. When a
lot of capital buys the same winners and sells the same losers, their residuals move
together. `crowding_score(stock_returns, long_members, short_members, factors=None,
window=52, method="leave_one_out", min_obs=26, min_stocks=5)` computes it separately for the
long and short legs at every formation date.

- **Membership** is two boolean DataFrames (formation dates by stocks) with identical
  formation dates. Formation can be less frequent than returns (monthly formation, weekly
  returns). Member stocks with no return column are ignored with a warning.
- **Window.** For formation date $\tau$, alphafade uses the last `window` rows of returns
  dated on or before $\tau$ (52 weeks, as in Lou & Polk). Nothing after $\tau$ is used, so the
  score is known at $\tau$. If fewer than `window` rows exist, the row is NaN (warm-up).
- **RF handling.** If `factors` has a column named `RF` (any case), it is subtracted from every
  stock return and is not used as a regressor. Factors must share the stock returns'
  frequency, cover every return date, and have no missing values.
- **Per-stock regressions.** Each stock's returns in the window are regressed on
  $[1, \text{factors}]$ (typically the Fama & French 1993 three factors) using only that
  stock's non-missing rows, and the residuals are kept.
  Without `factors`, the design is just a constant, so residuals are deviations from the
  stock's own mean and market-wide moves will dominate the score (not recommended).
- **min_obs and min_stocks.** A stock needs at least `min_obs` non-missing returns in the
  window (default 26, at least 3, at most `window`). A leg needs at least `min_stocks` usable
  stocks (default 5, at least 3), or its score is NaN. Both drops are counted in
  `DataDroppedWarning`s.

**Formulas.** Let $e_{i,t}$ be the residual of stock $i$ in period $t$, $N$ the number of
usable stocks in the leg, and $N_t$ the number with a residual in period $t$.

- `"leave_one_out"` (default, Lou & Polk): correlate each stock with the equal-weighted
  residual of the *other* stocks present that period,
  $\bar e_{-i,t} = \frac{1}{N_t - 1}\sum_{j \ne i} e_{j,t}$, then average over stocks:
  $\text{CoMOM} = \frac{1}{N}\sum_i \text{corr}_t(e_{i,t}, \bar e_{-i,t})$.
- `"pairwise"`: average over all pairs,
  $\text{CoMOM}^{pair} = \frac{2}{N(N-1)}\sum_{i<j} \text{corr}_t(e_{i,t}, e_{j,t})$.

Correlations use rows where both series are present and need at least 3 of them. The output
has `long`, `short`, `mean` (average of the legs that are not NaN), and `n_long`, `n_short`
(usable stocks).

**Why leave-one-out values are larger.** A portfolio's residual is less noisy than a single
stock's, because idiosyncratic noise averages out. If every pair has correlation $\rho$ and
equal variance, a stock's correlation with the average of the other $N-1$ is
$\rho / \sqrt{\rho + (1-\rho)/(N-1)}$, which tends to $\sqrt{\rho}$ as $N$ grows and exceeds
$\rho$ for $0 < \rho < 1$. For example, $\rho = 0.04$ with 100 stocks gives about 0.18. The
two methods are on different scales, so never compare one with the other.

## 10. Data (`alphafade.datasets`)

- **Files** (Kenneth French's data library): `F-F_Research_Data_Factors_CSV.zip` and its
  `_weekly`/`_daily` variants for `load_ff3(freq)` (`M`, `W`, `D`; the Fama & French 1993
  factors `Mkt-RF`, `SMB`, `HML`, plus `RF`), and `F-F_Momentum_Factor_CSV.zip` and `_daily`
  for `load_momentum(freq)` (`M`, `D`; the winners-minus-losers momentum factor of Jegadeesh
  & Titman 1993 and Carhart 1997, as a Series named `UMD`). There is no weekly momentum file.
- **Parsing.** Only the first table is read, up to the first blank line; the annual table and
  copyright notice below it are ignored. Monthly dates (`YYYYMM`) become month-end dates;
  weekly and daily rows keep French's exact date.
- **Percent to decimal.** French publishes percent, so every value is divided by 100 (2.89
  becomes 0.0289). Missing-data codes -99.99 and -999 become NaN with a `DataDroppedWarning`.
- **Caching.** Nothing touches the network at import time. The first call downloads the zip
  with the standard library (`urllib`, 30-second timeout) into `cache_dir=`, else
  `$ALPHAFADE_CACHE`, else `~/.cache/alphafade/`. The write is atomic (temp file, then
  rename). Later calls read the cache; `refresh=True` downloads again.
- **Offline.** `path=` loads a local `.zip` or `.csv` in French's format with no network or
  cache.

## 11. Known limitations

- **Survivorship bias.** If your universe holds only stocks that exist today, delisted stocks
  (often after crashing) are missing. That usually flatters a signal in early years, which can
  fake a decay, and distorts the short leg of the crowding score most. Use a point-in-time
  universe (for example CRSP) when you can.
- **A single break only.** The break tests look for one shift in the mean. A gradual decline
  shows up as one break somewhere in its middle, and several breaks are not separated. Use
  `fit_decay` to describe a smooth fade.
- **Asymptotic p-values.** The $\chi^2(1)$, sup-Wald, Kolmogorov (CUSUM), and Newey-West
  results are large-sample approximations; in short or very persistent series the true
  false-alarm rate can differ from the nominal one.
- **Decay toward zero.** $a e^{-\lambda t}$ has no floor. A signal that settles at a smaller
  but positive long-run level gets a longer half-life than its initial fall suggests.
- **Bootstrap percentile intervals** are simple but can be off when the bootstrap
  distribution is skewed or biased, as rate estimates near zero often are. Bootstrap rates
  that hit a grid bound are kept at the bound.
- **sup-Wald power loss.** Estimating the variance once from the whole demeaned series keeps
  the false-alarm rate honest, but a real break inflates that variance. In simulations at
  $n = 240$, a 1-standard-deviation shift was found 100% of the time either way, but a
  0.3-standard-deviation shift about 38% of the time, versus about 51% for the per-date
  variance (whose false-alarm rate is inflated to about 9%).

## 12. References

- Andrews, D. W. K. (1993). Tests for parameter instability and structural change with
  unknown change point. *Econometrica*, 61(4), 821-856.
- Andrews, D. W. K. (2003). Tests for parameter instability and structural change with
  unknown change point: A corrigendum. *Econometrica*, 71(1), 395-397.
- Carhart, M. M. (1997). On persistence in mutual fund performance. *Journal of Finance*,
  52(1), 57-82.
- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and
  bonds. *Journal of Financial Economics*, 33(1), 3-56.
- Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers:
  Implications for stock market efficiency. *Journal of Finance*, 48(1), 65-91.
- Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations.
  *Annals of Statistics*, 17(3), 1217-1241.
- Lou, D., & Polk, C. (2022). Comomentum: Inferring arbitrage activity from return
  correlations. *Review of Financial Studies*, 35(7), 3272-3302.
- McLean, R. D., & Pontiff, J. (2016). Does academic research destroy stock return
  predictability? *Journal of Finance*, 71(1), 5-32.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity
  and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708.
- Newey, W. K., & West, K. D. (1994). Automatic lag selection in covariance matrix
  estimation. *Review of Economic Studies*, 61(4), 631-653.
- Ploberger, W., & Krämer, W. (1992). The CUSUM test with OLS residuals. *Econometrica*,
  60(2), 271-285.
