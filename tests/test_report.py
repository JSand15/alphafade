from __future__ import annotations

import sys

import matplotlib
import numpy as np
import pandas as pd
import pytest

import alphafade as af

matplotlib.use("Agg")

IDX = pd.date_range("1965-01-31", "2014-12-31", freq="ME")
T = np.arange(len(IDX)) / 12


def strategy(seed: int = 0, tau: float = 12.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(0.012 * np.exp(-T / tau) + rng.normal(0, 0.01, len(IDX)), index=IDX)


def panel(seed: int = 0, k: int = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rho = 0.10 * np.exp(-T / 8)
    sig = rng.standard_normal((len(IDX), k))
    fwd = rho[:, None] * sig + np.sqrt(1 - rho[:, None] ** 2) * rng.standard_normal((len(IDX), k))
    return pd.DataFrame(sig, index=IDX), pd.DataFrame(fwd, index=IDX)


@pytest.fixture(scope="module")
def full_report() -> af.FadeReport:
    r = strategy()
    sig, fwd = panel()
    # Crowding that rises over time and predicts lower future returns.
    crowd = pd.Series(T / T[-1] + np.random.default_rng(1).normal(0, 0.05, len(IDX)), index=IDX)
    return af.analyze(
        r,
        signal=sig,
        fwd_returns=fwd,
        sample_end="1989-12-31",
        publication_date="1993-03-01",
        crowding=crowd,
        n_boot=200,
        rng=0,
    )


def test_all_sections_present(full_report: af.FadeReport) -> None:
    rep = full_report
    assert rep.freq == "M"
    assert rep.window == 36
    assert rep.return_decay.decay_detected
    assert rep.ic_decay is not None
    assert rep.ic_decay.decay_detected
    assert rep.ic_decay.half_life_years == pytest.approx(8 * np.log(2), rel=0.35)
    assert rep.publication is not None
    assert rep.publication_test is None
    assert rep.crowding_link is not None
    assert rep.crowding_link.slope_per_sd < 0
    assert rep.crowding_link.t_stat < -2
    assert rep.crowding_link.horizon == 12
    assert rep.rolling_ic is not None


def test_summary_and_verdict_are_plain_english(full_report: af.FadeReport) -> None:
    text = full_report.summary()
    for heading in (
        "Verdict:",
        "Decay of the average return",
        "Decay of the information coefficient",
        "Structural break",
        "Publication gap",
        "Crowding",
    ):
        assert heading in text
    verdict = full_report.verdict()
    assert verdict.startswith("Yes: the signal's IC is fading")
    assert "lower after publication" in verdict
    assert "Crowding has predicted weaker returns" in verdict


def test_to_frame_is_tidy(full_report: af.FadeReport) -> None:
    df = full_report.to_frame()
    assert list(df.columns) == ["section", "metric", "value"]
    assert set(df["section"]) == {
        "return_decay",
        "ic_decay",
        "break_test",
        "publication",
        "crowding_link",
    }
    hl = df.query("section == 'ic_decay' and metric == 'half_life_years'")["value"].iloc[0]
    assert hl == full_report.ic_decay.half_life_years  # type: ignore[union-attr]


def test_plot_returns_figure_with_all_panels(full_report: af.FadeReport) -> None:
    fig = full_report.plot()
    assert len(fig.axes) == 4
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_minimal_report_and_publication_only_date() -> None:
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.005, 0.02, len(IDX)), index=IDX)
    rep = af.analyze(r, publication_date="1993-03-01", n_boot=200, rng=3)
    assert rep.publication is None
    assert rep.publication_test is not None
    assert rep.ic is None
    assert rep.crowding_link is None
    assert "No detectable decay" in rep.verdict()
    assert "Change at the publication date" in rep.summary()
    assert set(rep.to_frame()["section"]) == {"return_decay", "break_test", "publication_test"}
    fig = rep.plot(figsize=(6, 4))
    assert len(fig.axes) == 2


def test_crowding_dataframe_and_link_validation() -> None:
    r = strategy()
    crowd_df = pd.DataFrame(
        {"long": 0.1, "short": 0.2, "mean": np.linspace(0, 1, len(IDX))}, index=IDX
    )
    rep = af.analyze(r, crowding=crowd_df, crowding_horizon=6, n_boot=200, rng=0)
    assert rep.crowding_link is not None
    assert rep.crowding_link.horizon == 6
    assert "Crowding" in rep.crowding_link.summary()
    with pytest.raises(af.InputError, match="'mean' column"):
        af.analyze(r, crowding=crowd_df.drop(columns="mean"), n_boot=200)
    with pytest.raises(af.InputError, match="crowding_horizon"):
        af.analyze(r, crowding=crowd_df, crowding_horizon=0, n_boot=200)
    with pytest.raises(af.InsufficientDataError, match="same dates"):
        af.analyze(r, crowding=crowd_df.iloc[:10], n_boot=200)
    with pytest.raises(af.InputError, match="constant"):
        af.analyze(r, crowding=crowd_df.assign(mean=1.0), n_boot=200)


def test_crowding_link_not_significant_wording() -> None:
    rng = np.random.default_rng(8)
    r = pd.Series(rng.normal(0.005, 0.02, len(IDX)), index=IDX)
    crowd = pd.Series(rng.normal(size=len(IDX)), index=IDX)
    rep = af.analyze(r, crowding=crowd, n_boot=200, rng=8)
    assert rep.crowding_link is not None
    assert "not a statistically reliable predictor" in rep.crowding_link.summary()


def test_input_validation() -> None:
    r = strategy()
    sig, _ = panel()
    with pytest.raises(af.InputError, match="both signal and fwd_returns"):
        af.analyze(r, signal=sig)
    with pytest.raises(af.InputError, match="publication_date too"):
        af.analyze(r, sample_end="1989-12-31")
    with pytest.raises(af.InsufficientDataError):
        af.analyze(pd.Series(np.nan, index=IDX))


def test_edge_nans_trimmed_and_custom_window() -> None:
    r = strategy()
    r.iloc[:5] = np.nan
    rep = af.analyze(r, window=24, n_boot=200, rng=0)
    assert rep.window == 24
    assert rep.returns.index[0] == IDX[5]


def test_plot_without_matplotlib_gives_install_hint(
    full_report: af.FadeReport, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match=r"alphafade\[plot\]"):
        full_report.plot()


def test_import_has_no_heavy_side_effects() -> None:
    import subprocess

    code = (
        "import sys, alphafade; "
        "bad = [m for m in ('matplotlib', 'urllib.request', 'statsmodels') if m in sys.modules]; "
        "print(bad)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
