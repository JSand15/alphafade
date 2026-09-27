# alphafade build spec (original brief from Jeevun, 2026-09-26)

This is the source-of-truth brief. Decisions that refine or override it are logged in
PROGRESS.md under "Decisions".

<about_me>
I'm Jeevs, a high school student who builds open-source quant finance and small-business
tools. My first library, finlearn-analytics, is live on PyPI with around 700 downloads.
I want this library to be genuinely useful to real people and to show careful,
research-grade engineering.
I'm still learning, so explain things in plain language. The first time you use a technical
term, define it in one sentence. If I share my finlearn-analytics repo, match its conventions
wherever they don't conflict with this spec.
</about_me>

<mission>
Build `alphafade`, a Python library that answers one question: "Is my trading signal dying, and
if so, how fast and why?"
Most tools measure how a signal's predictive power fades over the days after each trade (the
forecast horizon). alphafade measures something different: how a signal's edge shrinks across
calendar time (months and years), and whether that shrinkage lines up with crowding, meaning more
money chasing the same pattern. It is the software companion to my research on the
"Predictability Paradox": as more AI and quant traders exploit a pattern, the pattern weakens.
</mission>

<who_it_is_for>
- Student and academic researchers studying whether published market anomalies decay.
- Independent quants who want an early warning before a strategy stops paying.
- Anyone reproducing papers like McLean & Pontiff (2016), which found anomaly returns are about
  26% lower out-of-sample and about 58% lower after publication.
</who_it_is_for>

<in_scope>
1. Rolling signal quality: rolling information coefficient (IC) between a signal panel and
   next-period returns (Spearman by default, Pearson optional), and rolling Sharpe of a strategy
   return series.
2. Decay fitting: fit an exponential decay curve to rolling performance over calendar time and
   report the half-life (how many years until the edge is half gone), with a bootstrap
   confidence interval and goodness of fit. Also fit a linear trend as a simpler comparison and
   report which fits better.
3. Break detection: test whether performance changed at a known date (Chow-style test) and
   search for an unknown break date (CUSUM or sup-F), reporting the most likely break.
4. Publication gap: split a return series into in-sample, post-sample/pre-publication, and
   post-publication periods; report mean return, Sharpe, and the percent decline for each,
   with t-stats that use Newey-West standard errors.
5. Crowding score: a "comomentum"-style measure (Lou & Polk) - the average pairwise correlation
   of risk-adjusted returns among stocks in the long leg and among stocks in the short leg,
   computed over rolling windows. Rising correlation suggests more capital crowding the trade.
6. A single report object that bundles all of the above with .summary() (plain-English text),
   .to_frame(), and .plot().
</in_scope>

<out_of_scope>
- Not a backtester. The user supplies returns or a signal plus forward returns.
- Not a factor-construction library. Don't build portfolios from raw fundamentals.
- Not a replacement for alphalens. Don't reimplement forecast-horizon IC decay or tear sheets.
- No live data feeds or trading.
</out_of_scope>

<core_concepts>
- Information coefficient (IC): the correlation between today's signal values across assets and
  the next period's returns. Higher means the signal ranks winners and losers better.
- Half-life: if performance follows p(t) = a * exp(-t / tau), the half-life is tau * ln(2).
  If the fitted decay is not significantly different from zero, report "no detectable decay"
  rather than a huge meaningless half-life.
- Newey-West standard errors: t-stats that stay honest when returns are autocorrelated, which
  rolling and overlapping windows always create.
- Comomentum: for each window, regress each stock's weekly returns on a factor model (default
  Fama-French 3), take the residuals, and average pairwise residual correlations within each leg.
</core_concepts>

<public_api_sketch>
```python
import alphafade as af
ic = af.rolling_ic(signal, fwd_returns, window=36, method="spearman")  # signal: date x asset
perf = af.rolling_sharpe(strategy_returns, window=252)
fit = af.fit_decay(perf)  # -> DecayFit(half_life_years, ci_low, ci_high, r2, model)
brk = af.find_break(strategy_returns)  # -> BreakResult(date, stat, p_value)
gap = af.publication_gap(strategy_returns, sample_end="1993-12-31",
                         publication_date="1997-03-01")  # -> GapResult
crowd = af.crowding_score(stock_returns, long_members, short_members,
                          factors=ff3_weekly, window=52)
report = af.analyze(strategy_returns, signal=signal, fwd_returns=fwd_returns,
                    publication_date="1997-03-01")
print(report.summary())
report.plot()
```
Treat this as a sketch. Propose improvements in Milestone 0.
</public_api_sketch>

<package_layout>
```
src/alphafade/
  __init__.py      public API exports
  _validate.py     input checks, index alignment, frequency inference
  rolling.py       rolling_ic, rolling_sharpe
  decay.py         fit_decay, DecayFit
  breaks.py        find_break, test_break_at, BreakResult
  publication.py   publication_gap, GapResult
  crowding.py      crowding_score
  report.py        analyze, FadeReport
  plotting.py      optional, imported lazily (needs [plot] extra)
  datasets.py      optional loaders (Ken French library) with caching
tests/ examples/ docs/
```
</package_layout>

<dependencies>
Core: numpy, pandas, scipy, statsmodels (for Newey-West and regressions).
Extras: [plot] -> matplotlib. [data] -> requests (for Ken French downloads).
</dependencies>

<correctness_traps>
- Look-ahead: the IC at date t must pair the signal known at t with returns realized after t.
  Build alignment helpers and test them, because this is where most bugs hide.
- Overlapping windows create autocorrelation, so plain OLS t-stats on rolling series are
  overstated. Use Newey-West or block bootstrap.
- Exponential fits can explode or return nonsense on noisy data. Constrain parameters, use
  sensible starting values, and fall back to the linear model with a warning if the fit fails.
- Survivorship: if the user passes only currently listed stocks, warn in the docs that crowding
  and IC results will be biased.
- Mixed frequencies (daily vs monthly) must be detected and handled explicitly, never guessed.
</correctness_traps>

<validation_tests>
- Synthetic signal with IC exactly decaying as 0.10 * exp(-t / 5 years): fit_decay recovers a
  half-life within 10% of 5 * ln(2) years on long samples.
- Constant-IC synthetic signal: fit_decay reports no significant decay.
- Series with a planted mean shift at a known date: find_break locates it within a small
  tolerance.
- Crowding: simulate stocks with a common factor whose loading rises over time; crowding_score
  must trend upward. With independent stocks, it must stay near zero.
- Hand-computed IC on a tiny 3-asset, 4-date example matches exactly.
</validation_tests>

<milestones>
- M0 Plan, open questions, layout, API (no code).
- M1 Input validation and alignment helpers + rolling_ic, rolling_sharpe.
- M2 fit_decay with bootstrap CIs and linear fallback.
- M3 Break detection and publication_gap with Newey-West t-stats.
- M4 crowding_score.
- M5 analyze() report, plotting, Ken French loader, README, example on the UMD momentum factor.
- M6 CI, packaging, TestPyPI release.
</milestones>

<environment>
- macOS on Apple Silicon. Give terminal commands for macOS (zsh), not Windows.
- Use uv for virtual environments and dependencies. If it's missing: brew install uv
- Python >= 3.10.
- Git for version control, GitHub for hosting, PyPI for releases.
</environment>

<engineering_standards>
These exist because a library other people depend on has to be predictable and honest.
- Layout: src/ layout, pyproject.toml, hatchling build backend.
- Types: full type hints. mypy --strict must pass on the package (tests can be looser).
- Style: ruff for lint and format.
- Tests: pytest. Aim for >= 90% coverage on the core math modules. Use hypothesis for
  property-based tests where it fits naturally.
- Docs: NumPy-style docstring on every public function, each with a short runnable example.
- Dependencies: keep the core install small. Heavy or optional packages go in extras
  (for example: pip install alphafade[plot]).
- Network: no network calls at import time. Any download is explicit, cached under
  ~/.cache/alphafade/, and every function also accepts user-supplied data so it works offline.
- Randomness: every function that uses randomness takes a seed or numpy Generator argument,
  so results are reproducible.
- Errors: raise specific exceptions whose message tells the user how to fix their input.
  Never drop data silently. Emit a warning whenever you do something lossy (e.g. dropping NaNs).
- Public API: small and stable, exported from the top-level package. Everything else is
  prefixed with _ or lives in a private module.
- Project files: MIT license, CHANGELOG.md (Keep a Changelog format), semantic versioning.
</engineering_standards>

<how_to_work_with_me>
1. Plan before code. First, restate the goal in 3-5 sentences, list your open questions, and
   propose the package layout and public API. Then stop and wait for my approval.
2. Build one milestone at a time. For each one: write tests for the math first using synthetic
   data with known answers, implement, run the full test suite, run ruff and mypy, then stop.
3. Only call a milestone done after you have actually run the tests and they pass. Paste the
   pytest summary line as proof.
4. If a requirement is ambiguous, or you see a better design, ask me instead of guessing. If you
   disagree with part of this spec, tell me and explain why.
5. Keep PROGRESS.md at the repo root up to date: what's done, what's next, and each decision with
   its reason. A future session should be able to resume from that file alone.
6. Make a git commit after each milestone with a clear message.
7. Don't add features outside the scope below without asking me first.
</how_to_work_with_me>

<milestone_report_format>
After each milestone, reply with:
- What I built (2-4 bullets, plain language)
- Test results (pass/fail counts and coverage %)
- Decisions I made and why
- Anything that worries me
- The next milestone I propose
</milestone_report_format>

<definition_of_done_v0_1>
- pip install alphafade works in a fresh environment on macOS.
- README contains: a one-paragraph pitch, install command, a quickstart under 15 lines that runs
  exactly as written, a table of the public API, a "How this differs from alphalens and
  quantstats" section, and an honest Limitations section.
- examples/ has at least one end-to-end script or notebook on real or realistic data.
- GitHub Actions CI runs tests, ruff and mypy on Python 3.10-3.13, on macOS and Ubuntu.
- A release workflow publishes to TestPyPI first, then to PyPI when I push a version tag, using
  PyPI trusted publishing (no API tokens stored in the repo).
</definition_of_done_v0_1>

<first_task>
Start with Milestone 0 only: restate the goal, ask your open questions, and propose the layout
and public API. Do not write any code until I approve the plan.
</first_task>
