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
To be filled in at M1 once pyproject.toml exists (planned: uv sync, uv run pytest, uv run ruff,
uv run mypy).
