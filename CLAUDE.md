# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is
`alphafade`: a Python library that measures whether a trading signal's edge decays across
calendar time, how fast (half-life), and whether that lines up with crowding (comomentum).

## Read first, every session
1. `PROGRESS.md`: current milestone, next action, decisions log. It overrides `SPEC.md`
   where they differ.
2. `SPEC.md`: the original brief (scope, standards, validation tests, definition of done).

## Working rules (from the spec)
- Milestone-gated: build one milestone, then stop and report in the spec's
  milestone_report_format. Don't start the next milestone without Jeevun's approval.
- Per milestone: write tests with known-answer synthetic data first, then implement, then run
  the full pytest suite, ruff, and mypy --strict. Paste the pytest summary line as proof.
  Update PROGRESS.md, then commit.
- Ask before adding anything outside scope. Ask when a requirement is ambiguous.
- Jeevun is learning: plain language, and define each technical term the first time it
  appears.

## Commands
```zsh
uv sync                                   # create .venv with dev tools (Python from .python-version)
uv run pytest -q                          # full suite incl. doctests in src/
uv run pytest tests/test_rolling.py::test_hand_computed_spearman_ic   # one test
uv run pytest --cov --cov-report=term     # coverage
uv run ruff format . && uv run ruff check .
uv run mypy                               # --strict on src/alphafade (configured in pyproject)
uv run pytest -q -m network               # tests that download from Ken French (deselected by default)
uv run python scripts/make_supwald_table.py   # regenerate src/alphafade/_supwald_table.py (seeded)
uv run --extra plot python examples/umd_momentum.py   # end-to-end example on real data
UV_PROJECT_ENVIRONMENT=.venvs/py3.11 uv run --python 3.11 pytest -q   # another Python version
```
Releasing: see RELEASING.md (tag `vX.Y.Z` → TestPyPI → PyPI via trusted publishing).

## Architecture notes
- Public API = exactly what `src/alphafade/__init__.py` exports. Private helpers live in `_*.py`.
- All input coercion/validation goes through `_validate.py` (DatetimeIndex rules, frequency
  inference, alignment, NaN trimming). Don't reimplement checks in feature modules.
- Newey-West regression, lag rules, block bootstrap and RNG handling live in `_stats.py` and
  are verified against statsmodels (a dev-only dependency; never import it from src/).
- Rolling outputs carry `attrs["alphafade_window"]`; downstream code uses it to widen bootstrap
  blocks and Newey-West lags for overlapping windows.
- Every lossy step warns with `DataDroppedWarning`; tests treat unexpected alphafade warnings as
  errors (autouse fixture in `tests/conftest.py`), so use `pytest.warns` when one is expected.
- `analyze()` (report.py) only orchestrates: it calls the public functions and never
  re-implements their math. `plotting.py` is imported lazily from `FadeReport.plot()`.
- `datasets.py` is the only module allowed to touch the network (lazy `urllib` import).
- README code is executed by `tests/test_readme.py`. If you change README examples, the
  printed output shown in the README must still match exactly.
- pandas 3 may store datetimes at non-ns resolution: convert through `_validate._as_ns`, never
  `.asi8` directly.
