# Contributing to alphafade

Thanks for your interest! alphafade aims to be small, honest, and research-grade. Contributions
are welcome, especially bug reports with a reproduction, validation against published results,
and clearer documentation.

## Development setup

```bash
git clone https://github.com/JSand15/alphafade.git
cd alphafade
uv sync --extra plot        # creates .venv with the dev tools (brew install uv if needed)
uv run pytest -q            # tests + doctests (network tests are deselected)
uv run pytest -q -m network # the few tests that download from Ken French's site
uv run ruff format . && uv run ruff check .
uv run mypy                 # --strict on src/alphafade
```

## Ground rules

- **Keep the core small.** Runtime dependencies are numpy, pandas and scipy only. matplotlib
  lives in the `plot` extra. statsmodels is a *test-only* oracle: never import it from `src/`.
- **Validation lives in `src/alphafade/_validate.py`.** Reuse its helpers instead of re-checking
  inputs in each module. Never reindex, resample, or shift user data silently.
- **Errors say how to fix the input.** Raise the specific classes in `_errors.py`. Anything
  lossy emits a `DataDroppedWarning` with counts. The test suite fails on any alphafade warning
  a test doesn't expect, so use `pytest.warns` when you expect one.
- **Statistics come with known-answer tests.** New math needs a synthetic test where the right
  answer is known (and, where possible, a comparison to statsmodels or a published table).
- **Randomness is seeded.** Anything random takes `rng=` (int seed or `numpy.random.Generator`).
- **Public API = `alphafade/__init__.py`.** Everything else is private (`_`-prefixed modules
  or names). Every public function has a NumPy-style docstring with a runnable example.

## Regenerating the sup-Wald table

`src/alphafade/_supwald_table.py` is generated. To change the simulation, edit
`scripts/make_supwald_table.py` and run `uv run python scripts/make_supwald_table.py`
(about 10 seconds; seeded, so it's reproducible).

## Pull requests

1. Branch from `main`.
2. Make the change with tests; `pytest`, `ruff` and `mypy` must be clean.
3. Add a line to `CHANGELOG.md` under "Unreleased".
4. Open a PR. CI runs on Python 3.11 to 3.14, on Ubuntu and macOS, plus a job with the oldest
   supported numpy/pandas/scipy.
