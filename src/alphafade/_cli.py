"""The ``alphafade`` command: run a report on a CSV file without writing any Python.

This module only reads files and prints results. All the math lives in the public functions.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from ._errors import AlphaFadeError, AlphaFadeWarning, InputError
from .floor import fit_floor_decay
from .report import analyze

_PLOT_SUFFIXES = (".png", ".pdf", ".svg", ".jpg", ".jpeg")

# Date styles with only one possible reading: (pattern, strptime format, is a whole month).
_DATE_STYLES = (
    (r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d", False),
    (r"\d{4}-\d{2}-\d{2}[ T]00:00(:00)?", "%Y-%m-%d", False),
    (r"\d{4}-\d{2}", "%Y-%m", True),
    (r"\d{8}", "%Y%m%d", False),
    (r"\d{6}", "%Y%m", True),
)

_EXAMPLES = """\
examples:
  alphafade report returns.csv
  alphafade report returns.csv --sample-end 1989-12-31 --publication-date 1993-03-01
  alphafade report returns.csv --percent --plot chart.png
  alphafade floor ic.csv --column ic --json

The file is a comma-separated CSV with a header row: one date column and a column of numbers.
Returns are decimals (0.05 means 5%); add --percent if your file says 5 for 5%.
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line tool. Returns the exit code: 0 on success, 2 on bad input."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return 2
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", AlphaFadeWarning)
        try:
            output = _run_report(args) if args.command == "report" else _run_floor(args)
        except (AlphaFadeError, ImportError, OSError) as exc:
            _print_warnings(caught)
            print(f"error: {exc}", file=sys.stderr)
            return 2
    _print_warnings(caught)
    print(output)
    return 0


def _print_warnings(caught: list[warnings.WarningMessage]) -> None:
    # Warnings go to stderr, so `--json > out.json` still produces a clean file.
    for message in dict.fromkeys(str(w.message) for w in caught):
        print(f"warning: {message}", file=sys.stderr)


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("file", help="CSV file with a date column and a column of numbers")
    common.add_argument("--column", help="which column to analyse (if there is more than one)")
    common.add_argument("--date-column", help="which column holds the dates (default: the first)")
    common.add_argument(
        "--date-format",
        help="how the dates are written, e.g. %%d/%%m/%%Y (only needed for styles other than "
        "YYYY-MM-DD, YYYY-MM, YYYYMMDD and YYYYMM)",
    )
    common.add_argument(
        "--percent", action="store_true", help="values are percents (5 means 5%%): divide by 100"
    )
    common.add_argument(
        "--seed", type=int, default=0, help="random seed, so results repeat (default: 0)"
    )
    common.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        metavar="N",
        help="resamples for confidence intervals, at least 100 (default: 1000)",
    )
    common.add_argument("--json", action="store_true", help="print JSON instead of text")

    style = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(
        prog="alphafade",
        description="Is my trading signal dying? Run alphafade on a CSV file.",
        epilog=_EXAMPLES,
        formatter_class=style,
    )
    parser.add_argument("--version", action="version", version=f"alphafade {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="command")

    report = commands.add_parser(
        "report",
        parents=[common],
        help="full report on a strategy's returns",
        description="Full report on a strategy's per-period returns.",
        epilog=_EXAMPLES,
        formatter_class=style,
    )
    report.add_argument("--sample-end", help="last date of the original study's sample")
    report.add_argument("--publication-date", help="date the idea was published")
    report.add_argument("--freq", help="D, W, M, Q or A (default: worked out from the dates)")
    report.add_argument("--window", type=int, help="rolling window in periods")
    report.add_argument(
        "--plot",
        metavar="FILE",
        help="also save a chart (.png, .pdf, .svg or .jpg; needs alphafade[plot])",
    )

    commands.add_parser(
        "floor",
        parents=[common],
        help="does the edge fade to zero, or level off?",
        description="Does the edge fade all the way to zero, or level off at a floor? Works on "
        "any performance series: returns or a per-date IC.",
        epilog=_EXAMPLES,
        formatter_class=style,
    )
    return parser


def _run_report(args: argparse.Namespace) -> str:
    if args.plot is not None:
        _prepare_plotting(args.plot)  # fail before the slow part, not after it
    returns = _load(args)
    largest = float(returns.abs().max())
    if not args.percent and largest > 1:
        print(
            f"warning: the largest value is {largest:.4g}, which as a decimal return means "
            f"{largest:.0%}. If the file is in percents, add --percent.",
            file=sys.stderr,
        )
    report = analyze(
        returns,
        sample_end=args.sample_end,
        publication_date=args.publication_date,
        freq=args.freq,
        window=args.window,
        n_boot=args.bootstrap,
        rng=args.seed,
    )
    output = report.to_json() if args.json else report.summary()
    if args.plot is not None:
        figure = report.plot()
        figure.savefig(args.plot, dpi=150)
        print(f"chart saved to {args.plot}", file=sys.stderr)
    return output


def _run_floor(args: argparse.Namespace) -> str:
    fit = fit_floor_decay(_load(args), n_boot=args.bootstrap, rng=args.seed)
    if args.json:
        return json.dumps(fit.to_dict(), indent=2, allow_nan=False)
    return fit.summary()


def _prepare_plotting(target: str) -> None:
    if Path(target).suffix.lower() not in _PLOT_SUFFIXES:
        raise InputError(f"--plot needs a file ending in .png, .pdf, .svg or .jpg, got {target}.")
    try:
        import matplotlib
    except ImportError:
        raise ImportError(
            "--plot needs matplotlib. Install it with `pip install 'alphafade[plot]'`."
        ) from None
    matplotlib.use("Agg")  # draw to a file; never try to open a window


def _load(args: argparse.Namespace) -> pd.Series[float]:
    """Read one dated column of numbers from the CSV, refusing to guess when unsure."""
    if args.seed < 0:
        raise InputError(f"--seed must be 0 or more, got {args.seed}.")
    table = _read_csv(args.file)
    columns = [str(c) for c in table.columns]
    date_col = args.date_column if args.date_column is not None else columns[0]
    _require_column(date_col, columns, "--date-column")
    if args.column is not None:
        _require_column(args.column, columns, "--column")
        value_col = args.column
    else:
        others = [c for c in columns if c != date_col]
        numeric = [c for c in others if _is_numeric(table[c])]
        if len(numeric) > 1:
            raise InputError(
                f"{args.file} has several number columns ({', '.join(numeric)}). Pick one with "
                "--column."
            )
        if not numeric and len(others) != 1:
            raise InputError(
                f"No column of numbers found in {args.file} (columns: {', '.join(columns)}). "
                "Pick one with --column."
            )
        # With a single candidate column, use it so bad cells get reported by row.
        value_col = numeric[0] if numeric else others[0]
    if value_col == date_col:
        raise InputError(f"The column {value_col!r} can't be both the dates and the values.")
    values = _parse_numbers(table[value_col], value_col)
    if args.percent:
        values = values / 100.0
    dates = _parse_dates(table[date_col], args.date_format)
    return pd.Series(values.to_numpy(), index=dates, name=value_col)


def _read_csv(path: str) -> pd.DataFrame:
    try:
        with open(path, encoding="utf-8-sig", newline="") as handle, warnings.catch_warnings():
            # pandas only *warns* when a ragged row makes it throw values away: refuse instead.
            warnings.simplefilter("error", pd.errors.ParserWarning)
            table = pd.read_csv(handle, dtype=str, index_col=False, skipinitialspace=True)
    except pd.errors.EmptyDataError:
        raise InputError(f"{path} is empty.") from None
    except UnicodeDecodeError:
        raise InputError(
            f"{path} isn't a text file. alphafade reads CSV files; export spreadsheets as CSV."
        ) from None
    except (pd.errors.ParserError, pd.errors.ParserWarning, ValueError) as exc:
        raise InputError(f"Couldn't read {path} as a CSV file: {exc}") from None
    if table.shape[1] < 2:
        raise InputError(
            f"{path} has only one column. It needs a date column and a number column, "
            "separated by commas."
        )
    if table.empty:
        raise InputError(f"{path} has a header but no data rows.")
    return table


def _require_column(name: str, columns: list[str], flag: str) -> None:
    if name not in columns:
        raise InputError(f"{flag}: no column called {name!r}. Columns: {', '.join(columns)}.")


def _is_numeric(column: pd.Series[str]) -> bool:
    present = column.dropna()
    return bool(len(present)) and bool(pd.to_numeric(present, errors="coerce").notna().all())


def _first_bad(column: pd.Series[str], bad: np.ndarray) -> tuple[int, str]:
    """File line number (the header is line 1) and text of the first flagged cell."""
    position = int(np.argmax(bad))
    return position + 2, str(column.iloc[position])


def _parse_numbers(column: pd.Series[str], name: str) -> pd.Series[float]:
    numbers = pd.to_numeric(column, errors="coerce")
    bad = (numbers.isna() & column.notna()).to_numpy()
    if bad.any():
        row, text = _first_bad(column, bad)
        raise InputError(
            f"Column {name!r}, row {row}: {text!r} is not a number. Remove symbols such as % "
            "or thousands separators (for percents, use --percent)."
        )
    return numbers.astype("float64")


def _parse_dates(column: pd.Series[str], date_format: str | None) -> pd.DatetimeIndex:
    missing = column.isna().to_numpy()
    if missing.any():
        row = int(np.argmax(missing)) + 2
        raise InputError(f"The date is blank in row {row}.")
    text = column.str.strip()
    month_only = False
    if date_format is None:
        for pattern, fmt, whole_month in _DATE_STYLES:
            matches = text.str.fullmatch(pattern)
            if matches.iloc[0]:
                if not matches.all():
                    row, value = _first_bad(text, (~matches).to_numpy())
                    raise InputError(
                        f"Dates are written in more than one style (row {row}: {value!r}). "
                        "Make them consistent, or describe them with --date-format."
                    )
                date_format, month_only = fmt, whole_month
                text = text.str.slice(0, 10) if fmt == "%Y-%m-%d" else text
                break
        else:
            raise InputError(
                f"Can't read the date {text.iloc[0]!r} without guessing (day first or month "
                "first?). Say how dates are written, e.g. --date-format %d/%m/%Y."
            )
    bad_format = (
        f"--date-format {date_format!r} is not a usable date pattern. Use codes such as "
        "%d (day), %m (month) and %Y (year), e.g. %d/%m/%Y."
    )
    if "%" not in date_format:
        raise InputError(bad_format)
    # A pattern with a month but no day names a whole month, dated at its end like YYYY-MM.
    if any(c in date_format for c in ("%m", "%b", "%B")) and "%d" not in date_format:
        month_only = True
    try:
        dates = pd.to_datetime(text, format=date_format, errors="coerce")
    except ValueError:
        raise InputError(bad_format) from None
    bad = dates.isna().to_numpy()
    if bad.any():
        row, value = _first_bad(text, bad)
        raise InputError(
            f"Row {row}: the date {value!r} does not match {date_format} or doesn't exist "
            "(see --date-format)."
        )
    index = pd.DatetimeIndex(dates)
    if month_only:
        index = index + pd.offsets.MonthEnd(0)
    return index
