"""The `alphafade` command: same numbers as the library, and clear errors."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

import alphafade as af
from alphafade._cli import main
from tests.conftest import month_ends

matplotlib.use("Agg")


def returns_series(n: int = 480, seed: int = 0) -> pd.Series:
    t = np.arange(n) / 12
    noise = np.random.default_rng(seed).normal(0, 0.01, n)
    return pd.Series(0.004 + 0.012 * np.exp(-t * math.log(2) / 5) + noise, index=month_ends(n))


def write(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.fixture
def csv(tmp_path: Path) -> str:
    path = tmp_path / "returns.csv"
    returns_series().rename("ret").to_csv(path, index_label="date")
    return str(path)


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    try:
        code = main(list(argv))
    except SystemExit as exc:  # argparse exits by itself for --help and usage errors
        code = int(exc.code or 0)
    out, err = capsys.readouterr()
    return code, out, err


def fail(capsys: pytest.CaptureFixture[str], *argv: str) -> str:
    """Run a command that must fail cleanly; return its error text."""
    code, out, err = run(capsys, *argv)
    assert code == 2
    assert out == ""
    assert "error:" in err
    assert "Traceback" not in err
    return err


def floor_json(capsys: pytest.CaptureFixture[str], path: str, *extra: str) -> dict[str, object]:
    code, out, err = run(capsys, "floor", path, "--json", "--bootstrap", "100", *extra)
    assert code == 0, err
    return json.loads(out)  # type: ignore[no-any-return]


# ----------------------------------------------------------------------------- basics
def test_report_prints_the_library_summary(csv: str, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = run(capsys, "report", csv, "--bootstrap", "100")
    assert code == 0, err
    expected = af.analyze(returns_series(), n_boot=100, rng=0).summary()
    assert out.strip() == expected.strip()
    assert err == ""


def test_report_json_matches_library(csv: str, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(
        capsys,
        "report",
        csv,
        "--json",
        "--bootstrap",
        "100",
        "--seed",
        "7",
        "--sample-end",
        "1989-12-31",
        "--publication-date",
        "1993-03-01",
    )
    assert code == 0
    expected = af.analyze(
        returns_series(),
        n_boot=100,
        rng=7,
        sample_end="1989-12-31",
        publication_date="1993-03-01",
    ).to_dict()
    got = json.loads(out)
    assert got["publication"] is not None
    assert got["return_decay"]["half_life_years"] == pytest.approx(
        expected["return_decay"]["half_life_years"]
    )
    assert got["verdict"] == expected["verdict"]


def test_floor_matches_library(csv: str, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = run(capsys, "floor", csv, "--bootstrap", "200")
    assert code == 0, err
    assert out.strip() == af.fit_floor_decay(returns_series(), n_boot=200, rng=0).summary().strip()
    data = floor_json(capsys, csv)
    assert data["floor_outcome"] in {"survives", "may_reach_zero", "reverses", "unknown"}


def test_window_and_freq_are_passed_on(csv: str, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(
        capsys, "report", csv, "--json", "--bootstrap", "100", "--window", "24", "--freq", "M"
    )
    assert code == 0
    assert json.loads(out)["window"] == 24


def test_plot_writes_a_picture(
    csv: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "chart.png"
    code, _, err = run(capsys, "report", csv, "--bootstrap", "100", "--plot", str(target))
    assert code == 0, err
    assert target.read_bytes().startswith(b"\x89PNG")
    assert str(target) in err


def test_plot_json_keeps_stdout_clean(
    csv: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "chart.svg"
    code, out, _ = run(
        capsys, "report", csv, "--json", "--bootstrap", "100", "--plot", str(target)
    )
    assert code == 0
    json.loads(out)
    assert target.exists()


def test_help_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--version")
    assert code == 0
    assert out.strip() == f"alphafade {af.__version__}"
    code, out, _ = run(capsys, "--help")
    assert code == 0
    assert "report" in out
    assert "floor" in out
    code, out, _ = run(capsys, "report", "--help")
    assert code == 0
    assert "--percent" in out


def test_no_command_shows_help(capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run(capsys)
    assert code == 2
    assert "report" in err


def test_python_dash_m(csv: str) -> None:
    done = subprocess.run(
        [sys.executable, "-m", "alphafade", "floor", csv, "--bootstrap", "100"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "Floor decay fit" in done.stdout


def test_console_script_is_declared() -> None:
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert 'alphafade = "alphafade._cli:main"' in pyproject


# ----------------------------------------------------------------------------- reading CSVs
def test_percent_flag_divides_by_100(
    csv: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pct = tmp_path / "pct.csv"
    (returns_series() * 100).rename("ret").to_csv(pct, index_label="date")
    a = floor_json(capsys, csv)
    b = floor_json(capsys, str(pct), "--percent")
    assert b["aic_constant"] == pytest.approx(a["aic_constant"])
    assert b["p_value"] == a["p_value"]


def test_percent_looking_data_warns(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pct = tmp_path / "pct.csv"
    (returns_series() * 100).rename("ret").to_csv(pct, index_label="date")
    code, out, err = run(capsys, "report", str(pct), "--bootstrap", "100", "--json")
    assert code == 0
    assert "--percent" in err
    json.loads(out)


def test_floor_does_not_assume_values_are_returns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A Sharpe-like series legitimately exceeds 1; `floor` accepts any performance series.
    path = tmp_path / "sharpe.csv"
    (returns_series() * 200).rename("sharpe").to_csv(path, index_label="date")
    code, _, err = run(capsys, "floor", str(path), "--bootstrap", "100")
    assert code == 0
    assert "--percent" not in err


@pytest.mark.parametrize(
    "fmt", ["%Y-%m-%d", "%Y-%m", "%Y%m", "%Y%m%d", "%Y-%m-%d 00:00:00", "%Y-%m-%dT00:00"]
)
def test_unambiguous_date_styles(
    fmt: str, tmp_path: Path, csv: str, capsys: pytest.CaptureFixture[str]
) -> None:
    s = returns_series()
    path = tmp_path / "dates.csv"
    pd.DataFrame({"date": s.index.strftime(fmt), "ret": s.to_numpy()}).to_csv(path, index=False)
    assert floor_json(capsys, str(path)) == floor_json(capsys, csv)


@pytest.mark.parametrize("fmt", ["%b %Y", "%m/%Y"])
def test_month_only_custom_format_means_month_end(
    fmt: str, tmp_path: Path, csv: str, capsys: pytest.CaptureFixture[str]
) -> None:
    s = returns_series()
    path = tmp_path / "months.csv"
    pd.DataFrame({"date": s.index.strftime(fmt), "ret": s.to_numpy()}).to_csv(path, index=False)
    assert floor_json(capsys, str(path), "--date-format", fmt) == floor_json(capsys, csv)


def test_report_with_a_gap_warns_once(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    s = returns_series()
    s.iloc[100] = np.nan
    path = tmp_path / "gap.csv"
    s.rename("ret").to_csv(path, index_label="date")
    code, out, err = run(capsys, "report", str(path), "--bootstrap", "100", "--json")
    assert code == 0, err
    json.loads(out)
    lines = err.strip().splitlines()
    assert lines
    assert len(lines) == len(set(lines))
    assert all(line.startswith("warning: ") for line in lines)


def test_ambiguous_dates_need_a_format(
    tmp_path: Path, csv: str, capsys: pytest.CaptureFixture[str]
) -> None:
    s = returns_series()
    path = tmp_path / "us.csv"
    pd.DataFrame({"date": s.index.strftime("%m/%d/%Y"), "ret": s.to_numpy()}).to_csv(
        path, index=False
    )
    err = fail(capsys, "floor", str(path))
    assert "--date-format" in err
    assert "01/31/1980" in err
    assert floor_json(capsys, str(path), "--date-format", "%m/%d/%Y") == floor_json(capsys, csv)
    assert "does not match" in fail(capsys, "floor", str(path), "--date-format", "%d/%m/%Y")


@pytest.mark.parametrize("fmt", ["%q", "mixed", "ISO8601", "", "%"])
def test_bad_date_format(fmt: str, csv: str, capsys: pytest.CaptureFixture[str]) -> None:
    # "mixed" and "ISO8601" are pandas shortcuts that guess per row, so they are refused too.
    assert "--date-format" in fail(capsys, "floor", csv, "--date-format", fmt)


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("date,ret\n2020-02-30,0.1\n", "2020-02-30"),
        ("date,ret\n2020-01-31,0.1\n,0.2\n", "row 3"),
        ("date,ret\n2020-01-31,0.1\n202002,0.2\n", "--date-format"),
        ("date,ret\n202013,0.1\n", "202013"),
    ],
)
def test_bad_dates(
    body: str, match: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert match in fail(capsys, "floor", write(tmp_path / "x.csv", body))


def test_column_choice(tmp_path: Path, csv: str, capsys: pytest.CaptureFixture[str]) -> None:
    s = returns_series()
    path = tmp_path / "two.csv"
    pd.DataFrame(
        {"name": "x", "when": s.index.strftime("%Y-%m-%d"), "mine": s.to_numpy(), "other": 1.0}
    ).to_csv(path, index=False)
    err = fail(capsys, "floor", str(path), "--date-column", "when")
    assert "--column" in err
    assert "mine, other" in err
    picked = floor_json(capsys, str(path), "--date-column", "when", "--column", "mine")
    assert picked == floor_json(capsys, csv)
    err = fail(capsys, "floor", str(path), "--date-column", "when", "--column", "nope")
    assert "nope" in err
    assert "mine" in err
    assert "nope" in fail(capsys, "floor", str(path), "--date-column", "nope")


def test_text_columns_are_ignored_when_choosing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    s = returns_series()
    path = tmp_path / "t.csv"
    pd.DataFrame(
        {"date": s.index.strftime("%Y-%m-%d"), "label": "abc", "ret": s.to_numpy()}
    ).to_csv(path, index=False)
    floor_json(capsys, str(path))


def test_pandas_default_header_and_excel_bom(
    tmp_path: Path, csv: str, capsys: pytest.CaptureFixture[str]
) -> None:
    unnamed = tmp_path / "unnamed.csv"
    returns_series().to_csv(unnamed)  # header is ",0"
    assert floor_json(capsys, str(unnamed)) == floor_json(capsys, csv)
    bom = tmp_path / "bom.csv"
    bom.write_text("﻿" + Path(csv).read_text(), encoding="utf-8")
    assert floor_json(capsys, str(bom), "--date-column", "date") == floor_json(capsys, csv)


def test_non_numbers_are_named(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write(tmp_path / "x.csv", "date,ret\n2020-01-31,0.1\n2020-02-29,2.5%\n")
    err = fail(capsys, "floor", path)
    assert "2.5%" in err
    assert "row 3" in err


def test_blank_values_warn_on_stderr(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    s = returns_series()
    s.iloc[100] = np.nan
    path = tmp_path / "gap.csv"
    s.rename("ret").to_csv(path, index_label="date")
    code, out, err = run(capsys, "floor", str(path), "--bootstrap", "100", "--json")
    assert code == 0
    assert "warning: Dropped 1 missing" in err
    assert json.loads(out)["n_obs"] == 479


def test_unsorted_dates_are_refused_not_sorted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "rev.csv"
    returns_series().iloc[::-1].rename("ret").to_csv(path, index_label="date")
    assert "not sorted" in fail(capsys, "floor", str(path))


# ----------------------------------------------------------------------------- errors
@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("", "empty"),
        ("date,ret\n", "no data rows"),
        ("date;ret\n2020-01-31;0.1\n", "comma"),
        ('date,ret\n"2020-01-31,0.1\n', "read"),
        ("date,ret\n2020-01-31,0.1,5,6\n2020-02-29,0.1\n", "read"),
        ("date,ret\n2020-01-31,abc\n", "abc"),
        ("date,ret\n2020-01-31,0.1\n", "at least"),
    ],
)
def test_bad_files(
    body: str, match: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert match in fail(capsys, "floor", write(tmp_path / "x.csv", body))


def test_missing_file_and_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert "nothing.csv" in fail(capsys, "floor", str(tmp_path / "nothing.csv"))
    assert str(tmp_path) in fail(capsys, "floor", str(tmp_path))


def test_binary_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "book.xlsx"
    path.write_bytes(b"PK\x03\x04\x14\x00\xff\xfe\x00\x81")
    assert "CSV" in fail(capsys, "report", str(path))


def test_library_errors_become_messages(csv: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert "n_boot" in fail(capsys, "floor", csv, "--bootstrap", "5")
    assert "publication_date" in fail(capsys, "report", csv, "--sample-end", "1990-01-01")
    assert "not a valid date" in fail(
        capsys, "report", csv, "--publication-date", "soon", "--bootstrap", "100"
    )


def test_negative_seed(csv: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert "--seed" in fail(capsys, "floor", csv, "--seed", "-1")


@pytest.mark.parametrize("name", ["chart", "chart.csv", "chart.exe"])
def test_plot_needs_a_picture_extension(
    name: str, csv: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert ".png" in fail(capsys, "report", csv, "--plot", str(tmp_path / name))


def test_plot_into_missing_folder(
    csv: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "no" / "chart.png"
    assert "chart.png" in fail(capsys, "report", csv, "--bootstrap", "100", "--plot", str(target))


def test_plot_without_matplotlib(
    csv: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    err = fail(capsys, "report", csv, "--plot", str(tmp_path / "c.png"))
    assert "alphafade[plot]" in err
