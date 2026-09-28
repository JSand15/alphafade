"""Matplotlib charts for :class:`FadeReport`. Needs the ``plot`` extra.

This module is imported only when you call ``report.plot()``, so alphafade itself never
requires matplotlib.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .report import FadeReport

__all__ = ["plot_report"]

_INSTALL_HINT = (
    "Plotting needs matplotlib. Install it with `pip install 'alphafade[plot]'` "
    "(or `uv add 'alphafade[plot]'`)."
)


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return plt


def plot_report(report: FadeReport, figsize: tuple[float, float] | None = None) -> Figure:
    """Draw a :class:`FadeReport` as stacked panels sharing the date axis.

    Panels: cumulative return (with the break, sample end and publication dates marked),
    rolling Sharpe, the IC with its fitted decay (if a signal was given), and crowding (if
    given).
    """
    plt = _pyplot()
    panels = ["cumulative", "sharpe"]
    if report.ic is not None:
        panels.append("ic")
    if report.crowding is not None:
        panels.append("crowding")
    fig, axes = plt.subplots(
        len(panels),
        1,
        sharex=True,
        figsize=figsize or (10, 2.6 * len(panels) + 0.6),
        squeeze=False,
    )
    ax_list: list[Axes] = list(axes[:, 0])
    for ax, panel in zip(ax_list, panels, strict=True):
        {
            "cumulative": _cumulative,
            "sharpe": _sharpe,
            "ic": _ic,
            "crowding": _crowding,
        }[panel](ax, report)
        _mark_dates(ax, report)
        ax.grid(alpha=0.3)
    ax_list[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle(report.verdict(), fontsize=10, wrap=True)
    fig.tight_layout()
    return fig  # type: ignore[no-any-return]


def _cumulative(ax: Axes, report: FadeReport) -> None:
    growth = (1 + report.returns.fillna(0)).cumprod()
    ax.plot(growth.index, growth.to_numpy(), color="C0", lw=1.2, label="Growth of $1")
    if (growth > 0).all():
        ax.set_yscale("log")
    ax.set_ylabel("Growth of $1")
    fit = report.return_decay
    ax.set_title(
        f"Strategy returns: {fit.model} fit, "
        + (
            f"half-life {fit.half_life_years:.1f} yrs"
            if fit.half_life_years is not None
            else "no detectable decay"
        ),
        fontsize=9,
        loc="left",
    )


def _sharpe(ax: Axes, report: FadeReport) -> None:
    s = report.rolling_sharpe
    ax.plot(s.index, s.to_numpy(), color="C1", lw=1.2)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_ylabel(f"Rolling Sharpe\n({report.window} periods)")


def _ic(ax: Axes, report: FadeReport) -> None:
    assert report.ic is not None
    ic = report.ic
    ax.plot(ic.index, ic.to_numpy(), color="0.75", lw=0.6, label="IC per date")
    if report.rolling_ic is not None:
        ax.plot(
            report.rolling_ic.index,
            report.rolling_ic.to_numpy(),
            color="C2",
            lw=1.4,
            label="Rolling mean IC",
        )
    if report.ic_decay is not None:
        fitted = report.ic_decay.fitted
        ax.plot(
            fitted.index,
            fitted.to_numpy(),
            color="C3",
            lw=1.6,
            ls="--",
            label=f"{report.ic_decay.model} fit",
        )
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_ylabel("Information\ncoefficient")
    ax.legend(loc="upper right", fontsize=8, frameon=False)


def _crowding(ax: Axes, report: FadeReport) -> None:
    assert report.crowding is not None
    c = report.crowding
    ax.plot(c.index, c.to_numpy(), color="C4", lw=1.2)
    ax.set_ylabel("Crowding\n(comomentum)")


def _mark_dates(ax: Axes, report: FadeReport) -> None:
    marks: list[tuple[pd.Timestamp | None, str, str, str]] = [
        (
            report.break_test.date if report.break_test.significant() else None,
            "C3",
            "-.",
            "Estimated break",
        ),
        (report.sample_end, "0.45", "--", "Sample end"),
        (report.publication_date, "k", ":", "Publication"),
    ]
    for date, color, style, label in marks:
        if date is not None:
            # matplotlib's unit system converts Timestamps; its stubs just say float.
            ax.axvline(date, color=color, lw=1.1, ls=style, label=label)  # type: ignore[arg-type]
