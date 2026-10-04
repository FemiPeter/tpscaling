"""
Modulation figures: interaction strength, dR2 and predictor coverage, grouped
by scaling regime.

Two levels of aggregation:

``summarise_season_percentile``  one season and one percentile, as used in the
thesis figures.

``summarise_season``             pooled across percentiles for one season, as
used in the paper figures. Averages are taken within each percentile first and
then across percentiles, so a percentile with more grid cells does not dominate.

Scaling regimes come from the bivariate output of
:mod:`tpscaling.modulation`, which carries a ``scaling`` column derived from
the %/degC rate. Older bivariate files without that column are labelled on the
fly from ``temp_slope`` and ``temp_sd``; a file with neither cannot be used,
because classifying a standardised slope as if it were %/degC puts almost every
cell in the super-C-C category.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import (PERCENTILES, PREDICTOR_ORDER, REGIME_ORDER, SEASONS,
               normalize_regime)

#: Predictors below this coverage within a regime are not drawn.
MIN_COVERAGE_PERCENT = 2.0


def _imports():
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:                                # pragma: no cover
        raise ImportError("plotting requires matplotlib and seaborn") from exc
    return plt, sns


def load_pair(bivariate_file, delta_file) -> pd.DataFrame:
    """Join the bivariate regimes onto the per-predictor dR2 records."""
    from ..modulation import add_scaling_rate

    bivariate = pd.read_csv(bivariate_file)
    if "scaling" not in bivariate.columns:
        standardised = bool(bivariate.get("standardized", pd.Series([False])).iloc[0])
        if standardised and "temp_sd" in bivariate.columns:
            bivariate = add_scaling_rate(bivariate)
        elif not standardised and "temp_slope" in bivariate.columns:
            # An unstandardised bivariate fit already gives ln(P) per degC, so
            # the slope converts to %/degC directly.
            from ..scaling import categorize_slope
            bivariate = bivariate.copy()
            bivariate["temp_slope_percent_per_degC"] = bivariate["temp_slope"] * 100.0
            bivariate["scaling"] = (bivariate["temp_slope_percent_per_degC"]
                                    .map(categorize_slope))
        else:
            raise ValueError(
                f"{bivariate_file} has no 'scaling' column, and cannot be "
                "labelled: a standardised fit needs 'temp_sd' to convert to "
                "%/degC, an unstandardised one needs 'temp_slope'")

    delta = pd.read_csv(delta_file)
    bivariate = bivariate.copy()
    bivariate["scaling"] = bivariate["scaling"].map(normalize_regime)
    merged = delta.merge(bivariate[["grid_cell", "scaling"]], on="grid_cell",
                         how="inner")
    merged["modulation"] = merged["interaction_coefficient"] * 100.0
    merged["delta_r2_percent"] = merged["delta_r2"] * 100.0
    merged["significant"] = merged["interaction_pvalue"] < 0.05
    return merged


def _summarise(records: pd.DataFrame, regime_counts: dict, divisor: int,
               predictors, percentile_column=None) -> pd.DataFrame:
    rows = []
    for regime in REGIME_ORDER:
        total_cells = regime_counts.get(regime, 0)
        denominator = total_cells * divisor
        for predictor in predictors:
            subset = records[(records["scaling"] == regime)
                             & (records["predictor"] == predictor)]
            significant = subset[subset["significant"]]
            if percentile_column and len(subset):
                means = subset.groupby(percentile_column)[
                    ["modulation", "delta_r2_percent"]].mean().mean()
                modulation = float(means["modulation"])
                delta = float(means["delta_r2_percent"])
                if len(significant):
                    sig_means = significant.groupby(percentile_column)[
                        ["modulation", "delta_r2_percent"]].mean().mean()
                    modulation_sig = float(sig_means["modulation"])
                    delta_sig = float(sig_means["delta_r2_percent"])
                else:
                    modulation_sig = delta_sig = np.nan
            else:
                modulation = subset["modulation"].mean() if len(subset) else np.nan
                delta = (subset["delta_r2_percent"].mean() if len(subset)
                         else np.nan)
                modulation_sig = (significant["modulation"].mean()
                                  if len(significant) else np.nan)
                delta_sig = (significant["delta_r2_percent"].mean()
                             if len(significant) else np.nan)
            rows.append({
                "scaling_regime": regime, "predictor": predictor,
                "regime_cells": total_cells, "n_records": len(subset),
                "n_significant": int(significant.shape[0]),
                "coverage_percent": (100.0 * len(subset) / denominator
                                     if denominator else 0.0),
                "significant_percent": (100.0 * len(significant) / denominator
                                        if denominator else 0.0),
                "mean_modulation": modulation,
                "mean_modulation_significant": modulation_sig,
                "mean_delta_r2": delta,
                "mean_delta_r2_significant": delta_sig,
            })
    return pd.DataFrame(rows)


def summarise_season_percentile(modulation_dir, percentile: int,
                                predictors=PREDICTOR_ORDER) -> pd.DataFrame:
    """Summary for one season and one percentile."""
    modulation_dir = Path(modulation_dir)
    label = f"tau_{percentile:02d}.csv"
    records = load_pair(modulation_dir / "bivariate" / label,
                        modulation_dir / "change_in_r2" / label)
    counts = (pd.read_csv(modulation_dir / "bivariate" / label)
              .pipe(lambda d: d if "scaling" in d.columns
                    else __import__("tpscaling.modulation", fromlist=["x"])
                    .add_scaling_rate(d))["scaling"].value_counts().to_dict())
    summary = _summarise(records, counts, 1, predictors)
    summary.insert(0, "percentile", percentile)
    return summary


def summarise_season(modulation_dir, percentiles=PERCENTILES,
                     predictors=PREDICTOR_ORDER) -> pd.DataFrame:
    """Summary for one season, pooled across percentiles."""
    modulation_dir = Path(modulation_dir)
    frames, counts = [], {}
    for percentile in percentiles:
        label = f"tau_{percentile:02d}.csv"
        bivariate = modulation_dir / "bivariate" / label
        delta = modulation_dir / "change_in_r2" / label
        if not (bivariate.exists() and delta.exists()):
            continue
        records = load_pair(bivariate, delta)
        records["percentile"] = percentile
        frames.append(records)
        if not counts:
            counts = records.groupby("scaling")["grid_cell"].nunique().to_dict()
    if not frames:
        return pd.DataFrame()
    pooled = pd.concat(frames, ignore_index=True)
    return _summarise(pooled, counts, len(frames), predictors,
                      percentile_column="percentile")


def _panel(axis, summary, predictors, value_column, significant_column,
           colours, bar_width=0.15, stacked_coverage=False):
    positions = np.arange(len(REGIME_ORDER))
    drawn = [p for p in predictors
             if (summary[summary["predictor"] == p]["coverage_percent"]
                 > MIN_COVERAGE_PERCENT).any()]
    for index, predictor in enumerate(drawn):
        offset = positions + index * bar_width - len(drawn) * bar_width / 2
        values, significant = [], []
        for regime in REGIME_ORDER:
            row = summary[(summary["scaling_regime"] == regime)
                          & (summary["predictor"] == predictor)]
            if row.empty or row["coverage_percent"].iloc[0] <= MIN_COVERAGE_PERCENT:
                values.append(0.0); significant.append(np.nan)
            else:
                values.append(float(row[value_column].iloc[0]))
                significant.append(float(row[significant_column].iloc[0]))
        if stacked_coverage:
            lower = [v * s / 100.0 if np.isfinite(s) else 0.0
                     for v, s in zip(values, significant)]
            upper = [v - l for v, l in zip(values, lower)]
            axis.bar(offset, lower, bar_width, color=colours[predictor],
                     alpha=1.0, hatch="///")
            axis.bar(offset, upper, bar_width, bottom=lower,
                     color=colours[predictor], alpha=0.7)
        else:
            axis.bar(offset, values, bar_width, color=colours[predictor],
                     alpha=0.7, zorder=2)
            for x, value in zip(offset, significant):
                if np.isfinite(value) and value != 0:
                    axis.scatter(x, value, color=colours[predictor], s=70,
                                 marker="D", edgecolors="black", linewidth=1,
                                 zorder=5)
    axis.set_xticks(positions)
    axis.grid(True, alpha=0.3)
    return drawn


def plot_panels(summaries, quantity: str, output_file, titles=None,
                predictors=PREDICTOR_ORDER, ylabel=None, ylim=None):
    """Stack one panel per entry in ``summaries``.

    ``quantity`` is ``"modulation"``, ``"delta_r2"`` or ``"coverage"``.
    """
    plt, sns = _imports()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    colours = dict(zip(predictors, sns.color_palette("Set2", len(predictors))))
    columns = {
        "modulation": ("mean_modulation", "mean_modulation_significant",
                       "Mean interaction coefficient (\u00d7100)", (-50, 50)),
        "delta_r2": ("mean_delta_r2", "mean_delta_r2_significant",
                     "Mean \u0394R\u00b2 (%)", (0, 15)),
        "coverage": ("coverage_percent", "significant_percent",
                     "Grid cells (%)", (0, 115)),
    }
    if quantity not in columns:
        raise ValueError(f"unknown quantity {quantity!r}")
    value_column, significant_column, default_label, default_ylim = columns[quantity]

    n = len(summaries)
    fig, axes = plt.subplots(n, 1, figsize=(16, 5 * n), squeeze=False)
    axes = axes[:, 0]
    drawn = []
    for index, summary in enumerate(summaries):
        axis = axes[index]
        drawn = _panel(axis, summary, predictors, value_column,
                       significant_column, colours,
                       stacked_coverage=(quantity == "coverage"))
        axis.set_ylabel(ylabel or default_label, fontsize=18)
        axis.set_ylim(ylim or default_ylim)
        if quantity == "modulation":
            axis.axhline(0, color="black", linewidth=1, alpha=0.5, zorder=1)
        if titles:
            axis.set_title(titles[index], fontsize=22, fontweight="bold")
        axis.set_xticklabels([])
        axis.tick_params(axis="y", labelsize=18)
        axis.text(0.02, 0.98, chr(97 + index), transform=axis.transAxes,
                  fontsize=22, fontweight="bold", va="top", ha="left")

    axes[-1].set_xticklabels(list(REGIME_ORDER), fontsize=18)
    axes[-1].set_xlabel("Scaling regime", fontsize=18)

    handles = [Patch(facecolor=colours[p], alpha=0.7, label=p) for p in drawn]
    if quantity == "coverage":
        handles.append(Rectangle((0, 0), 1, 1, facecolor="white",
                                 edgecolor="black", hatch="///",
                                 label="p < 0.05"))
    else:
        handles.append(Line2D([0], [0], marker="D", color="w",
                              markerfacecolor="white", markeredgecolor="black",
                              markersize=10, label="p < 0.05"))
    axes[-1].legend(handles=handles, loc="upper center",
                    bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=18,
                    framealpha=1.0)

    fig.tight_layout()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_file}")
    return output_file
