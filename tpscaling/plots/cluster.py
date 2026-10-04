"""
Mean predictor temperature-sensitivity by scaling regime.

Two layouts, both one row per predictor and bars grouped by scaling regime:

``plot_by_percentile``  x axis is percentile, with a pooled ``Mean`` column
appended. One figure per season; used in the thesis.

``plot_by_season``      x axis is season, each value already pooled across
percentiles. Used in the paper.

Input is the table written by ``tpscaling cluster``
(``predictor_slopes_by_regime.csv``), which carries ``predictor``,
``percentile``, ``scaling_regime`` and ``mean_slope``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import (PREDICTOR_ORDER, PREDICTOR_UNITS, REGIME_COLOURS,
               REGIME_ORDER, normalize_regime)


def _imports():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:                                # pragma: no cover
        raise ImportError("plotting requires matplotlib") from exc
    return plt


def _draw(axes, available, categories, values_of, xlabel, titles):
    bar_width = 0.2
    positions = np.arange(len(categories))
    for index, predictor in enumerate(available):
        axis = axes[index]
        for offset, (regime, colour) in enumerate(REGIME_COLOURS.items()):
            values = [values_of(predictor, regime, category)
                      for category in categories]
            axis.bar(positions + offset * bar_width
                     - (len(REGIME_COLOURS) - 1) * bar_width / 2,
                     values, bar_width, label=regime, color=colour, alpha=0.8,
                     edgecolor="white", linewidth=0.5)
        axis.text(0.01, 0.98, chr(97 + index), transform=axis.transAxes,
                  fontsize=24, fontweight="bold", va="top")
        axis.set_title(titles(predictor), fontsize=19, fontweight="bold",
                       pad=10)
        axis.set_ylabel(PREDICTOR_UNITS.get(predictor, ""), fontsize=17)
        axis.grid(True, alpha=0.3, axis="y")
        axis.set_axisbelow(True)
        axis.set_xticks(positions)
        axis.tick_params(axis="y", labelsize=15)
        if index == len(available) - 1:
            axis.set_xticklabels(categories, fontsize=17)
            axis.set_xlabel(xlabel, fontsize=17)
            axis.tick_params(axis="x", labelsize=15)
        else:
            axis.set_xticklabels([""] * len(categories))
        axis.yaxis.offsetText.set_fontsize(14)


def _finish(fig, axes, output_file):
    plt = _imports()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.55, 0.02),
               ncol=4, fontsize=15, frameon=True, fancybox=True,
               title="Scaling regime", title_fontsize=17)
    fig.tight_layout()
    fig.subplots_adjust(top=0.94, bottom=0.11, hspace=0.2)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_file}")
    return output_file


def plot_by_percentile(cluster_file, output_file, season_label: str = "",
                       predictors=PREDICTOR_ORDER):
    """One panel per predictor; bars by percentile, plus a pooled mean."""
    plt = _imports()
    table = pd.read_csv(cluster_file)
    table["predictor"] = table["predictor"].replace({"VIMFC1": "VIMFC"})
    table["scaling_regime"] = table["scaling_regime"].map(normalize_regime)
    available = [p for p in predictors if p in set(table["predictor"])]
    percentiles = sorted(table["percentile"].unique())
    categories = [f"{p}th" for p in percentiles] + ["Mean"]

    lookup = {(r.predictor, r.scaling_regime, f"{r.percentile}th"): r.mean_slope
              for r in table.itertuples()}
    pooled = (table.groupby(["predictor", "scaling_regime"])["mean_slope"]
              .mean().to_dict())

    def values_of(predictor, regime, category):
        if category == "Mean":
            return pooled.get((predictor, regime), 0.0)
        return lookup.get((predictor, regime, category), 0.0)

    fig, axes = plt.subplots(len(available), 1,
                             figsize=(14, 4.2 * len(available)), squeeze=False)
    _draw(axes[:, 0], available, categories, values_of, "Percentile",
          lambda p: f"{season_label}-{p}".strip("-"))
    return _finish(fig, axes[:, 0], output_file)


def plot_by_season(cluster_files: dict, output_file,
                   predictors=PREDICTOR_ORDER):
    """One panel per predictor; bars by season, each pooled across percentiles.

    ``cluster_files`` maps season name to its cluster table.
    """
    plt = _imports()
    records = []
    for season, path in cluster_files.items():
        if not Path(path).exists():
            print(f"  missing cluster table for {season}: {path}")
            continue
        table = pd.read_csv(path)
        table["predictor"] = table["predictor"].replace({"VIMFC1": "VIMFC"})
        table["scaling_regime"] = table["scaling_regime"].map(normalize_regime)
        pooled = (table.groupby(["predictor", "scaling_regime"])["mean_slope"]
                  .mean().reset_index())
        pooled["season"] = season
        records.append(pooled)
    if not records:
        raise ValueError("no cluster tables found")
    combined = pd.concat(records, ignore_index=True)

    available = [p for p in predictors if p in set(combined["predictor"])]
    seasons = [s for s in cluster_files if s in set(combined["season"])]
    lookup = {(r.predictor, r.scaling_regime, r.season): r.mean_slope
              for r in combined.itertuples()}

    fig, axes = plt.subplots(len(available), 1,
                             figsize=(16, 4.2 * len(available)), squeeze=False)
    _draw(axes[:, 0], available, seasons,
          lambda p, r, s: lookup.get((p, r, s), 0.0), "Season", lambda p: p)
    return _finish(fig, axes[:, 0], output_file)
