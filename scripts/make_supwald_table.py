"""Simulate the asymptotic null distribution of the sup-Wald (Andrews 1993) break test.

For one tested parameter (a shift in the mean), the sup-Wald statistic converges under "no
break" to

    sup over pi in [trim, 1 - trim] of  B(pi)^2 / (pi * (1 - pi)),

where B is a Brownian bridge. This script simulates that distribution on a fine grid with a
fixed seed and writes quantiles to src/alphafade/_supwald_table.py, which alphafade uses to turn
a statistic into a p-value. Re-running it reproduces the file exactly.

Usage:  uv run python scripts/make_supwald_table.py
"""

from __future__ import annotations

import pathlib
import time

import numpy as np

SEED = 20260927
N_REPS = 100_000
N_STEPS = 5_000
CHUNK = 500
TRIMS = (0.05, 0.10, 0.15, 0.20, 0.25)
# CDF levels at which quantiles are stored: dense in the right tail, where p-values matter.
LEVELS = np.unique(
    np.concatenate(
        [
            np.linspace(0.01, 0.80, 80),
            np.linspace(0.80, 0.99, 96),
            np.linspace(0.99, 0.999, 19),
            np.array([0.9995]),
        ]
    ).round(6)
)


def simulate() -> dict[float, np.ndarray]:
    rng = np.random.default_rng(SEED)
    grid = np.arange(1, N_STEPS + 1) / N_STEPS
    stats = {trim: np.empty(N_REPS) for trim in TRIMS}
    masks = {trim: (grid >= trim) & (grid <= 1 - trim) for trim in TRIMS}
    for start in range(0, N_REPS, CHUNK):
        steps = rng.standard_normal((CHUNK, N_STEPS)) / np.sqrt(N_STEPS)
        w = np.cumsum(steps, axis=1)
        bridge = w - grid[None, :] * w[:, -1:]
        for trim in TRIMS:
            m = masks[trim]
            g = grid[m]
            stats[trim][start : start + CHUNK] = (bridge[:, m] ** 2 / (g * (1 - g))).max(axis=1)
    return stats


def main() -> None:
    t0 = time.time()
    stats = simulate()
    lines = [
        '"""Quantiles of the sup-Wald null distribution (one parameter). GENERATED FILE.',
        "",
        "Produced by scripts/make_supwald_table.py",
        f"(seed={SEED}, reps={N_REPS}, steps={N_STEPS}). Do not edit by hand.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        f"LEVELS: tuple[float, ...] = {tuple(float(x) for x in LEVELS)!r}",
        "",
        "QUANTILES: dict[float, tuple[float, ...]] = {",
    ]
    for trim in TRIMS:
        q = np.quantile(stats[trim], LEVELS)
        lines.append(f"    {trim}: {tuple(round(float(x), 4) for x in q)!r},")
    lines.append("}")
    out = pathlib.Path(__file__).resolve().parents[1] / "src" / "alphafade" / "_supwald_table.py"
    out.write_text("\n".join(lines) + "\n")
    for trim in TRIMS:
        cv = np.quantile(stats[trim], [0.90, 0.95, 0.99])
        print(f"trim={trim:.2f}  10%={cv[0]:.2f}  5%={cv[1]:.2f}  1%={cv[2]:.2f}")
    print(f"wrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
