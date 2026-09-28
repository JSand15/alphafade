"""Is momentum dying? An end-to-end alphafade run on the real UMD factor.

UMD ("up minus down") is Ken French's momentum factor: each month it buys the stocks that
went up most over the prior year (skipping the latest month) and shorts the ones that went
down most. Momentum was documented by Jegadeesh & Titman, whose sample ended in December
1989 and whose paper was published in the Journal of Finance in March 1993. That makes it a
good test case for McLean & Pontiff-style publication decay.

Run it (downloads a few KB once, then works offline from the cache):

    uv run --extra plot python examples/umd_momentum.py

Or, with a file you downloaded yourself:

    uv run --extra plot python examples/umd_momentum.py --path F-F_Momentum_Factor_CSV.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import alphafade as af

SAMPLE_END = "1989-12-31"  # end of Jegadeesh & Titman's sample
PUBLICATION = "1993-03-01"  # Journal of Finance, March 1993


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", help="local copy of F-F_Momentum_Factor_CSV.zip (offline)")
    parser.add_argument("--start", default="1963-07-31", help="first month to analyse")
    parser.add_argument("--out", default="examples/output", help="folder for the chart")
    args = parser.parse_args()

    umd = af.datasets.load_momentum("M", path=args.path)
    umd = umd.loc[args.start :]

    report = af.analyze(
        umd,
        sample_end=SAMPLE_END,
        publication_date=PUBLICATION,
        n_boot=2000,
        rng=42,
    )
    print(report.summary())
    print()
    print(report.to_frame().to_string(index=False))

    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        print("\n(Install the plot extra to also save a chart: pip install 'alphafade[plot]')")
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig = report.plot()
    fig.savefig(out / "umd_report.png", dpi=130)
    print(f"\nChart saved to {out / 'umd_report.png'}")


if __name__ == "__main__":
    main()
