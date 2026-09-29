"""The README's code must run exactly as written (and print what the README says)."""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

README = (Path(__file__).resolve().parents[1] / "README.md").read_text()
BLOCKS = re.findall(r"```(\w+)\n(.*?)```", README, flags=re.DOTALL)
PYTHON = [code for lang, code in BLOCKS if lang == "python"]


def run(code: str) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, "README.md", "exec"), {"__name__": "__readme__"})
    return out.getvalue()


def test_readme_has_quickstart_under_15_lines() -> None:
    quickstart = PYTHON[0]
    assert "af.analyze" in quickstart
    assert len(quickstart.strip().splitlines()) < 15


def test_signal_example_runs_and_prints_what_readme_shows() -> None:
    code = next(c for c in PYTHON if "ic_series" in c)
    printed = run(code)
    shown = next(c for lang, c in BLOCKS if lang == "text" and "Exponential decay fit" in c)
    assert printed.strip() == shown.strip()


@pytest.mark.network
def test_quickstart_runs_as_written() -> None:
    printed = run(PYTHON[0])
    assert "Verdict:" in printed
    assert "After publication" in printed


def test_beyond_the_basics_example_runs_and_prints_what_readme_shows() -> None:
    code = next(c for c in PYTHON if "signal_lifetime" in c)
    printed = run(code)
    shown = next(c for lang, c in BLOCKS if lang == "text" and "reaches 50%" in c)
    assert printed.strip() == shown.strip()
