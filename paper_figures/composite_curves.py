"""
Composite ln(P)-T curves by scaling category and relationship type.
Thesis Fig. 1-7 style, built from your own binned data.

For each season, percentile, scaling category and relationship type:
  - collect the grid cells (prefix = column_x) in that group
  - average the "<pct>th Percentile_log" column of each cell's precipitation
    bin-summary, bin by bin
  - average the "Mean" column of each cell's temperature bin-summary, bin by bin
  - plot mean ln(P) against mean T

Bins are aligned BY POSITION (row index), not by the Interval string, because
each grid cell has its own temperature-derived bin edges.

Edit CONFIG, then run:  python composite_curves.py
"""

import os
import glob
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- CONFIG
ROOT = r"F:\scaling_seasonal"

# folder name for each season -> ROOT\<folder>\
# EDIT THESE to match your actual folder names.
SEASON_DIRS = {
    "Annual":       "annual_s",
    "Winter (DJF)": "winter_s",
    "Spring (MAM)": "spring_s",
    "Summer (JJA)": "summer_s",
    "Autumn (SON)": "autumn_s",
}

PERCENTILES = [75, 90, 95, 99]

# Keep only cells whose scaling fit was statistically significant?
ONLY_SIGNIFICANT = False

# Require at least this many cells before a group is plotted
MIN_CELLS = 5

# Everything is written here; the folder is created if it does not exist.
OUTDIR = os.path.join(ROOT, "composite_curves_output")

OUT_CSV = "composite_curves_aggregated.csv"
OUT_FIG_FULL = "Fig_composite_curves_full.png"
OUT_FIG_ANNUAL = "Fig_composite_curves_annual.png"
DPI = 500

# ------------------------------------------------------------- CATEGORIES
SCALING_ORDER = ["Super C-C", "C-C", "Sub C-C", "Negative"]
REL_ORDER = ["monotonic", "peak_structure"]
REL_LABEL = {"monotonic": "Monotonic", "peak_structure": "Peak structure"}

# Okabe-Ito palette: distinct and colour-blind safe
PCT_COLOURS = {75: "#009E73", 90: "#E69F00", 95: "#D55E00", 99: "#0072B2"}

# Clausius-Clapeyron reference slope, in ln(P) per degC (7 % per degC)
CC_SLOPE = 0.07
SHOW_CC_REFERENCE = True
SHOW_TPEAK_VALUES = True


def normalise_scaling(s):
    """Map the messy scaling strings onto four canonical labels."""
    if not isinstance(s, str):
        return None
    t = s.lower()
    if "super" in t:
        return "Super C-C"
    if "sub" in t:
        return "Sub C-C"
    if "negative" in t:
        return "Negative"
    if "c-c" in t or "c–c" in t:      # 'C-C like scaling'
        return "C-C"
    return None


def normalise_rel(s):
    if not isinstance(s, str):
        return None
    t = s.strip().lower()
    if t.startswith("peak"):
        return "peak_structure"
    if t.startswith("mono"):
        return "monotonic"
    return None


def find_scaling_file(season_dir, pct):
    """Locate the P<pct> scaling csv inside a season folder."""
    for pat in (f"*P{pct}*.csv", f"*p{pct}*.csv", f"*{pct}*.csv"):
        hits = glob.glob(os.path.join(season_dir, pat))
        hits = [h for h in hits if os.path.isfile(h)]
        if hits:
            return sorted(hits, key=len)[0]
    return None


def cell_number(prefix):
    m = re.search(r"(\d+)", str(prefix))
    return int(m.group(1)) if m else None


def load_series(path, column):
    """Read one bin-summary file and return the requested column as an array."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if column not in df.columns:
        # tolerate stray whitespace in headers
        cmap = {c.strip(): c for c in df.columns}
        if column not in cmap:
            return None
        column = cmap[column]
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def stack_mean(arrays):
    """Average a list of equal-length arrays position-wise, ignoring NaN.

    Arrays of differing length are truncated to the most common length; any
    that are shorter are dropped."""
    arrays = [a for a in arrays if a is not None and len(a) > 0]
    if not arrays:
        return None, 0
    lengths = [len(a) for a in arrays]
    n_bins = int(pd.Series(lengths).mode().iloc[0])
    kept = [a[:n_bins] for a in arrays if len(a) >= n_bins]
    if not kept:
        return None, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.vstack(kept), axis=0), len(kept)


# ------------------------------------------------------------ AGGREGATION
def aggregate():
    rows = []
    for season, folder in SEASON_DIRS.items():
        sdir = os.path.join(ROOT, folder)
        prec_dir = os.path.join(sdir, "prec_bin", "bin_summary")
        temp_dir = os.path.join(sdir, "tmean_bin", "bin_summary")
        if not os.path.isdir(prec_dir):
            print(f"  ! missing {prec_dir} - skipping {season}")
            continue

        for pct in PERCENTILES:
            spath = find_scaling_file(sdir, pct)
            if spath is None:
                print(f"  ! no P{pct} scaling file in {sdir}")
                continue
            sc = pd.read_csv(spath)
            sc.columns = [c.strip() for c in sc.columns]
            sc["_scaling"] = sc["scaling"].map(normalise_scaling)
            sc["_rel"] = sc["relationship"].map(normalise_rel)
            sc["_cell"] = sc["prefix"].map(cell_number)
            if ONLY_SIGNIFICANT and "statistical_significance" in sc.columns:
                sig = sc["statistical_significance"].astype(str).str.upper()
                sc = sc[sig.isin(["TRUE", "1", "YES"])]

            pcol = f"{pct}th Percentile_log"

            for scat in SCALING_ORDER:
                for rel in REL_ORDER:
                    grp = sc[(sc._scaling == scat) & (sc._rel == rel)]
                    cells = [c for c in grp["_cell"].dropna().astype(int)]
                    if len(cells) < MIN_CELLS:
                        continue

                    p_arrays, t_arrays = [], []
                    for c in cells:
                        pf = os.path.join(
                            prec_dir, f"column_{c}_precipitation_result_summary.csv")
                        tf = os.path.join(
                            temp_dir, f"column_{c}_temperature_result_summary.csv")
                        if not (os.path.isfile(pf) and os.path.isfile(tf)):
                            continue
                        pa = load_series(pf, pcol)
                        ta = load_series(tf, "Mean")
                        if pa is None or ta is None:
                            continue
                        n = min(len(pa), len(ta))
                        p_arrays.append(pa[:n])
                        t_arrays.append(ta[:n])

                    pmean, npc = stack_mean(p_arrays)
                    tmean, ntc = stack_mean(t_arrays)
                    if pmean is None or tmean is None:
                        continue
                    n = min(len(pmean), len(tmean))

                    tpk = np.nan
                    if rel == "peak_structure" and "peak_temp" in grp.columns:
                        tpk = pd.to_numeric(grp["peak_temp"],
                                            errors="coerce").mean()

                    kmax = int(np.nanargmax(pmean[:n])) if n > 2 else 0
                    tpeak_of_mean = float(tmean[kmax]) if n > 2 else np.nan

                    for i in range(n):
                        rows.append({
                            "season": season, "percentile": f"P{pct}",
                            "scaling": scat, "relationship": rel,
                            "bin_index": i,
                            "mean_temperature_C": tmean[i],
                            "mean_logP": pmean[i],
                            "n_cells": min(npc, ntc),
                            "Tpeak_mean_of_cells_C": tpk,
                            "Tpeak_of_composite_C": tpeak_of_mean,
                        })
            print(f"  {season} P{pct}: done")

    out = pd.DataFrame(rows)
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, OUT_CSV)
    out.to_csv(path, index=False)
    print(f"\nWrote {path}  ({len(out)} rows)")
    return out


# ---------------------------------------------------------------- PLOTTING
def _panel(ax, df, scat, rel, season, show_x, show_y):
    sub = df[(df.scaling == scat) & (df.relationship == rel) &
             (df.season == season)]
    peaks = []                      # (percentile, T_peak) for the corner block
    for pct in PERCENTILES:
        d = sub[sub.percentile == f"P{pct}"].sort_values("bin_index")
        if d.empty:
            continue
        T = d.mean_temperature_C.to_numpy()
        Y = d.mean_logP.to_numpy()
        ax.plot(T, Y, "-", lw=1.1, color=PCT_COLOURS[pct], label=f"P{pct}",
                zorder=4)

        # faint C-C reference: anchored at the curve's coldest bin, 7 % per degC.
        # Truncated to the data's own y-range so it cannot inflate the y-limits.
        if SHOW_CC_REFERENCE and len(T) > 1:
            ref = Y[0] + CC_SLOPE * (T - T[0])
            lo, hi = Y.min() - 0.15, Y.max() + 0.25
            keep = (ref >= lo) & (ref <= hi)
            if keep.sum() > 1:
                ax.plot(T[keep], ref[keep], "--", lw=0.65,
                        color=PCT_COLOURS[pct], alpha=0.38, zorder=2)

        # PEAK OF THE MEAN: the maximum of the composite curve itself, rather
        # than the average of the individual cells' peak temperatures.
        if rel == "peak_structure" and len(Y) > 2 and np.isfinite(Y).any():
            k = int(np.nanargmax(Y))
            interior = 0 < k < len(Y) - 1
            # An edge maximum is not a peak: the composite is still rising (or
            # falling) at the end of the sampled range. Shown hollow and in grey
            # so it is never mistaken for a genuine turning point.
            ax.plot([T[k]], [Y[k]], "o", ms=3.6, zorder=6, mew=0.5,
                    mfc=PCT_COLOURS[pct] if interior else "white",
                    mec="0.2" if interior else "0.55")
            peaks.append((pct, T[k], interior))

    # T_peak values as a block in the top-left corner, not on the curves
    if SHOW_TPEAK_VALUES and peaks:
        ax.text(0.035, 0.975, "$T_{\\mathrm{peak}}$ (\u00b0C)",
                transform=ax.transAxes, ha="left", va="top", fontsize=5.4,
                color="0.35")
        for i, (pct, tv, interior) in enumerate(peaks):
            ax.text(0.035, 0.885 - 0.085 * i,
                    f"P{pct}  {tv:.1f}" + ("" if interior else "*"),
                    transform=ax.transAxes, ha="left", va="top", fontsize=5.4,
                    color=PCT_COLOURS[pct] if interior else "0.55")

    ax.tick_params(labelsize=6, length=2)
    ax.spines[["top", "right"]].set_visible(False)
    # NB: with shared axes, set_xticklabels([]) would blank every panel,
    # because shared axes share one formatter. Toggle label visibility instead.
    ax.tick_params(labelbottom=show_x, labelleft=show_y)


def plot_full(df):
    seasons = [s for s in SEASON_DIRS if s in set(df.season)]
    combos = [(sc, r) for sc in SCALING_ORDER for r in REL_ORDER]
    fig, axes = plt.subplots(len(combos), len(seasons),
                             figsize=(7.09, 9.4), sharex="col", sharey=True,
                             squeeze=False)
    for i, (scat, rel) in enumerate(combos):
        for j, season in enumerate(seasons):
            _panel(axes[i, j], df, scat, rel, season,
                   show_x=(i == len(combos) - 1), show_y=(j == 0))
            if i == 0:
                axes[i, j].set_title(season, fontsize=8.5, pad=4)
        axes[i, 0].set_ylabel(f"{scat}\n{REL_LABEL[rel]}", fontsize=7)
    fig.supxlabel("Temperature (\u00b0C)", fontsize=8.5, y=0.028)
    fig.supylabel("ln(extreme precipitation)", fontsize=8.5, x=0.005)
    h, l = axes[0, 0].get_legend_handles_labels()
    if SHOW_CC_REFERENCE:
        from matplotlib.lines import Line2D
        h = h + [Line2D([], [], ls="--", lw=0.9, color="0.45")]
        l = l + ["C\u2013C reference (7 % $^{\\circ}$C$^{-1}$)"]
    fig.legend(h, l, ncol=5, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.016), fontsize=8)
    fig.tight_layout(rect=[0.015, 0.048, 1, 1])
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, OUT_FIG_FULL)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_annual(df, season="Annual"):
    """Condensed 2x4 version, annual only - the manuscript-sized figure."""
    fig, axes = plt.subplots(2, 4, figsize=(7.09, 3.6), sharex=True,
                             sharey=True, squeeze=False)
    for j, scat in enumerate(SCALING_ORDER):
        for i, rel in enumerate(REL_ORDER):
            ax = axes[i, j]
            _panel(ax, df, scat, rel, season, show_x=(i == 1), show_y=(j == 0))
            ax.set_title(f"({chr(97 + i * 4 + j)}) {scat}, {REL_LABEL[rel]}",
                         fontsize=7.5, pad=3)
    fig.supxlabel("Temperature (\u00b0C)", fontsize=8.5, y=-0.04)
    fig.supylabel("ln(extreme precipitation)", fontsize=8.5, x=0.005)
    h, l = axes[0, 0].get_legend_handles_labels()
    if SHOW_CC_REFERENCE:
        from matplotlib.lines import Line2D
        h = h + [Line2D([], [], ls="--", lw=0.9, color="0.45")]
        l = l + ["C\u2013C reference (7 % $^{\\circ}$C$^{-1}$)"]
    fig.legend(h, l, ncol=5, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.16), fontsize=8)
    fig.tight_layout(rect=[0.015, 0.02, 1, 1])
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, OUT_FIG_ANNUAL)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    data = aggregate()
    if not data.empty:
        plot_full(data)
        plot_annual(data)
