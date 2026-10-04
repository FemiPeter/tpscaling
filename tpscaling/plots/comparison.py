"""
Percentile binning versus quantile regression.

A scatter matrix of the two scaling-rate estimates, one panel per
season-percentile pair, with the Pearson correlation annotated. The layout is
**percentiles as rows, seasons as columns**, matching the manuscript figure.

The x axis is the percentile-binning estimate (``slope2`` from the stage 4
scaling tables, in %/degC) and the y axis is the quantile-regression estimate
(``full_slope_percent`` from ``tpscaling quantreg``).

Both estimates are in %/degC, so the one-to-one line is meaningful. Do not pass
the standardised slopes from :mod:`tpscaling.modulation` here: those are per
standard deviation of temperature and will not lie near the diagonal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import PERCENTILES, SEASONS


def _imports():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:                                # pragma: no cover
        raise ImportError("plotting requires matplotlib") from exc
    return plt


def load_pair(binning_file, quantreg_file) -> pd.DataFrame | None:
    """Merge the two estimates on grid-cell identifier."""
    binning = pd.read_csv(binning_file)
    quantile = pd.read_csv(quantreg_file)
    if "prefix" not in binning.columns or "grid_cell" not in quantile.columns:
        return None
    merged = binning.merge(quantile, left_on="prefix", right_on="grid_cell",
                           how="inner")
    merged = merged[["prefix", "slope2", "full_slope_percent"]].dropna()
    return merged if len(merged) else None


def plot_matrix(binning_dirs: dict, quantreg_dirs: dict, output_file,
                seasons=SEASONS, percentiles=PERCENTILES,
                show_one_to_one: bool = True, dpi: int = 300):
    """Scatter matrix with percentiles as rows and seasons as columns.

    ``binning_dirs`` maps season to the stage 4 ``scaling`` directory;
    ``quantreg_dirs`` maps season to the ``quantreg`` output directory.
    """
    plt = _imports()

    seasons = [s for s in seasons if s in binning_dirs and s in quantreg_dirs]
    fig, axes = plt.subplots(len(percentiles), len(seasons),
                             figsize=(4.4 * len(seasons),
                                      4.0 * len(percentiles)), squeeze=False)

    for row, percentile in enumerate(percentiles):
        for column, season in enumerate(seasons):
            axis = axes[row, column]
            binning_file = Path(binning_dirs[season]) / f"P{percentile}.csv"
            quantreg_file = (Path(quantreg_dirs[season])
                             / f"tau_{percentile:02d}.csv")
            merged = (load_pair(binning_file, quantreg_file)
                      if binning_file.exists() and quantreg_file.exists()
                      else None)

            if merged is None:
                axis.text(0.5, 0.5, f"No data\n{season} P{percentile}",
                          ha="center", va="center", transform=axis.transAxes,
                          fontsize=12)
            else:
                x = merged["slope2"].to_numpy()
                y = merged["full_slope_percent"].to_numpy()
                r, _ = stats.pearsonr(x, y)
                axis.scatter(x, y, alpha=0.7, s=18, edgecolors="white",
                             linewidth=0.4)
                fit = np.poly1d(np.polyfit(x, y, 1))
                grid = np.linspace(x.min(), x.max(), 100)
                axis.plot(grid, fit(grid), "r--", linewidth=1.4, alpha=0.85)
                if show_one_to_one:
                    low = min(x.min(), y.min())
                    high = max(x.max(), y.max())
                    axis.plot([low, high], [low, high], color="0.5", lw=0.8,
                              ls=":", zorder=0)
                axis.text(0.95, 0.05, f"$r = {r:.3f}$",
                          transform=axis.transAxes, fontsize=18,
                          va="bottom", ha="right",
                          bbox=dict(boxstyle="round", facecolor="white",
                                    alpha=0.8))

            axis.set_title(f"{season} \u2013 P{percentile}", fontsize=18,
                           fontweight="bold")
            axis.grid(True, alpha=0.3, linewidth=0.5)
            axis.tick_params(labelsize=14)
            if column == 0:
                axis.set_ylabel("Quantile regression (% $^{\\circ}$C$^{-1}$)",
                                fontsize=15)
            if row == len(percentiles) - 1:
                axis.set_xlabel("Percentile binning (% $^{\\circ}$C$^{-1}$)",
                                fontsize=15)

    fig.tight_layout(pad=2.0)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {output_file}")
    return output_file
