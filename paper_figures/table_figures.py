"""
Matrix-style replacements for Table 1 and Table 2, plus a labelled-bar
alternative for Table 1.

Outputs (PNG only, 500 dpi):
  Fig_coverage_matrix.png        - Table 1 as an annotated matrix
  Fig_coverage_bars.png          - Table 1 as stacked bars, all values labelled
  Fig_peak_matrix.png            - Table 2 as an annotated matrix
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "savefig.dpi": 500,
})

PERIODS = ["Annual", "Winter", "Spring", "Summer", "Autumn"]
PLABEL = {"Annual": "Annual", "Winter": "Winter (DJF)", "Spring": "Spring (MAM)",
          "Summer": "Summer (JJA)", "Autumn": "Autumn (SON)"}
PCTS = ["P75", "P90", "P95", "P99"]
NP = len(PERIODS) * len(PCTS)
SEP = [3.5, 7.5, 11.5, 15.5]
CENTRES = [1.5, 5.5, 9.5, 13.5, 17.5]

# scaling-category colours, matched to the maps in Fig. 2
C_NEG, C_SUB, C_CC, C_SUPER = "#FF9377", "#90C8FD", "#659BFC", "#1B4FA8"

t1 = pd.read_excel("scaling_tables.xlsx", sheet_name="Table1_S1_scaling")
t2 = pd.read_excel("scaling_tables.xlsx", sheet_name="TableS2_peak")


def ramp(name, dark):
    """White -> category colour ramp, for per-row shading."""
    return LinearSegmentedColormap.from_list(name, ["#FFFFFF", dark])


def season_labels(ax, y, size=8.5):
    for c, per in zip(CENTRES, PERIODS):
        ax.text(c, y, PLABEL[per], ha="center", va="top", fontsize=size,
                fontweight="bold")


def draw_row(ax, i, vals, cmap, vmax, fmt="{:.1f}", thresh=0.62):
    """Shade one matrix row with its own colormap and annotate every cell."""
    norm = Normalize(0, vmax)
    for j, v in enumerate(vals):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   facecolor=cmap(norm(v)), edgecolor="none"))
        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.4,
                color="white" if norm(v) > thresh else "black", zorder=3)


# =========================================================== TABLE 1: MATRIX
def coverage_matrix():
    order = ["Super C-C", "C-C", "Sub C-C", "Negative"]
    src = {"Super C-C": "Super C-C", "C-C": "C-C like",
           "Sub C-C": "Sub C-C", "Negative": "Negative"}
    cols = {"Super C-C": C_SUPER, "C-C": C_CC, "Sub C-C": C_SUB, "Negative": C_NEG}
    rlab = ["Super-C\u2013C (>9)", "C\u2013C (5\u20139)",
            "Sub-C\u2013C (0\u20135)", "Negative (<0)"]

    def series(cat, col):
        return np.array([t1[(t1.Period == p) & (t1.Category == src[cat]) &
                            (t1.Percentile == q)][col].values[0]
                         for p in PERIODS for q in PCTS])

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.09, 4.35), sharex=True)

    cov_cmap = ramp("cov", "#24528F")
    for i, cat in enumerate(order):
        draw_row(ax, i, series(cat, "Coverage_pct"), cov_cmap, 62, thresh=0.55)
    ax.set_xlim(-1.35, NP - 0.5); ax.set_ylim(3.5, -0.5)
    # category colour swatches beside the row labels
    for i, cat in enumerate(order):
        ax.add_patch(plt.Rectangle((-1.15, i - 0.28), 0.42, 0.56,
                                   facecolor=cols[cat], edgecolor="0.5",
                                   linewidth=0.4, clip_on=False, zorder=5))
    ax.set_yticks(range(4)); ax.set_yticklabels(rlab, fontsize=7.5)
    for k in SEP:
        ax.axvline(k, color="white", lw=2.4, zorder=4)
    ax.tick_params(length=0); ax.spines[:].set_visible(False)
    ax.set_title("(a) Areal coverage (% of TP grid cells)", fontsize=8.5,
                 loc="left", pad=4)

    rate = np.vstack([series(c, "MeanRate_pct_per_C") for c in order])
    norm = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=14)
    cmap = plt.get_cmap("RdBu_r")
    for i in range(4):
        for j in range(NP):
            v = rate[i, j]
            bx.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor=cmap(norm(v)), edgecolor="none"))
            bx.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.4,
                    color="white" if abs(v) > 8.5 else "black", zorder=3)
    bx.set_xlim(-1.35, NP - 0.5); bx.set_ylim(3.5, -0.5)
    for i, cat in enumerate(order):
        bx.add_patch(plt.Rectangle((-1.15, i - 0.28), 0.42, 0.56,
                                   facecolor=cols[cat], edgecolor="0.5",
                                   linewidth=0.4, clip_on=False, zorder=5))
    bx.set_yticks(range(4)); bx.set_yticklabels(rlab, fontsize=7.5)
    bx.set_xticks(range(NP))
    bx.set_xticklabels(PCTS * len(PERIODS), fontsize=7)
    for k in SEP:
        bx.axvline(k, color="white", lw=2.4, zorder=4)
    bx.tick_params(length=0); bx.spines[:].set_visible(False)
    bx.set_title("(b) Mean scaling rate (% $^{\\circ}$C$^{-1}$)", fontsize=8.5,
                 loc="left", pad=4)
    season_labels(bx, 4.55)

    fig.tight_layout(); fig.subplots_adjust(hspace=0.30, bottom=0.15)
    fig.savefig("Fig_coverage_matrix.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================ TABLE 1: BARS
def coverage_bars():
    order = ["Negative", "Sub C-C", "C-C like", "Super C-C"]
    cols = {"Negative": C_NEG, "Sub C-C": C_SUB, "C-C like": C_CC,
            "Super C-C": C_SUPER}
    fig, axes = plt.subplots(2, 5, figsize=(7.09, 4.5), sharey="row")
    x = np.arange(4)

    for j, per in enumerate(PERIODS):
        ax = axes[0, j]
        sub = t1[t1.Period == per]
        bottom = np.zeros(4)
        chips = []
        for cat in order:
            vals = np.array([sub[(sub.Category == cat) & (sub.Percentile == q)]
                             .Coverage_pct.values[0] for q in PCTS])
            ax.bar(x, vals, bottom=bottom, color=cols[cat], width=0.72,
                   edgecolor="white", linewidth=0.5)
            for xi, (v, b) in enumerate(zip(vals, bottom)):
                if v >= 6:
                    ax.text(xi, b + v / 2, f"{v:.1f}", ha="center", va="center",
                            fontsize=6.0,
                            color="white" if cat == "Super C-C" else "black")
                else:
                    chips.append((xi, v, cols[cat]))
            bottom += vals
        seen = {}
        for xi, v, c in chips:
            k = seen.get(xi, 0); seen[xi] = k + 1
            ax.text(xi, 105 + 8.5 * k, f"{v:.1f}", ha="center", va="center",
                    fontsize=5.8, zorder=6,
                    bbox=dict(facecolor=c, edgecolor="0.6", linewidth=0.3,
                              boxstyle="round,pad=0.22"))
        ax.set_xticks(x); ax.set_xticklabels(PCTS, fontsize=7)
        ax.set_ylim(0, 124); ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_xlim(-0.6, 3.6)
        ax.set_title(f"({chr(97 + j)}) {PLABEL[per]}", fontsize=8.5, pad=3)
        ax.tick_params(length=2)
        if j == 0:
            ax.set_ylabel("Areal coverage (%)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_bounds(0, 100)

    for j, per in enumerate(PERIODS):
        ax = axes[1, j]
        sub = t1[t1.Period == per]
        off = {"Super C-C": 6, "C-C like": -11, "Sub C-C": 6, "Negative": -11}
        for cat in order:
            vals = np.array([sub[(sub.Category == cat) & (sub.Percentile == q)]
                             .MeanRate_pct_per_C.values[0] for q in PCTS])
            ax.plot(x, vals, "-o", color=cols[cat], ms=3.2, lw=1.2,
                    mec="0.25", mew=0.4)
            for xi, v in zip(x, vals):
                ax.annotate(f"{v:.1f}", (xi, v), textcoords="offset points",
                            xytext=(0, off[cat]), ha="center", fontsize=5.4)
        ax.axhspan(5, 9, color="0.88", zorder=0, lw=0)
        ax.axhline(0, color="0.4", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(PCTS, fontsize=7)
        ax.set_ylim(-15, 19); ax.set_xlim(-0.6, 3.6)
        ax.set_title(f"({chr(102 + j)}) {PLABEL[per]}", fontsize=8.5, pad=3)
        ax.tick_params(length=2)
        if j == 0:
            ax.set_ylabel("Mean scaling rate (% $^{\\circ}$C$^{-1}$)")
        ax.spines[["top", "right"]].set_visible(False)

    h = [Patch(fc=cols[c], ec="white") for c in order]
    lab = ["Negative (<0 % $^{\\circ}$C$^{-1}$)", "Sub-C\u2013C (0\u20135)",
           "C\u2013C (5\u20139)", "Super-C\u2013C (>9)"]
    fig.legend(h, lab, ncol=4, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), handlelength=1.4, columnspacing=1.6)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.subplots_adjust(wspace=0.18, hspace=0.42)
    fig.savefig("Fig_coverage_bars.png", bbox_inches="tight")
    plt.close(fig)


# =========================================================== TABLE 2: MATRIX
def peak_matrix():
    rec = t2.set_index(["Period", "Percentile"])
    g = lambda c: np.array([rec.loc[(p, q), c] for p in PERIODS for q in PCTS])
    rows = [("Monotonic (+)", g("Monotonic_pos"), "mono"),
            ("Monotonic (\u2212)", g("Monotonic_neg"), "mono"),
            ("Peak structure (+)", g("Peak_pos"), "peak"),
            ("Peak structure (\u2212)", g("Peak_neg"), "peak")]
    tp = g("Tpeak_C")

    # distinct colour families: monotonic = teal/green, peak = purple/magenta
    cm_mono = ramp("mono", "#0F6E63")
    cm_peak = ramp("peak", "#6A2C86")

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.09, 3.75), sharex=True,
                                 gridspec_kw={"height_ratios": [4, 1.3]})

    for i, (_, vals, fam) in enumerate(rows):
        draw_row(ax, i, vals, cm_mono if fam == "mono" else cm_peak, 55)
    ax.set_xlim(-0.5, NP - 0.5); ax.set_ylim(3.5, -0.5)
    ax.set_yticks(range(4)); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    for k in SEP:
        ax.axvline(k, color="white", lw=2.4, zorder=4)
    ax.axhline(1.5, color="white", lw=2.4, zorder=4)
    ax.tick_params(length=0); ax.spines[:].set_visible(False)
    for lab, y, c in [("Monotonic", 0.5, "#0F6E63"), ("Peak", 2.5, "#6A2C86")]:
        ax.text(-0.75, y, "", color=c)
    ax.set_title("(a) Scaling structure (% of TP grid cells)", fontsize=8.5,
                 loc="left", pad=4)

    norm = TwoSlopeNorm(vmin=-9, vcenter=0, vmax=11)
    cmap = plt.get_cmap("RdBu_r")
    for j, v in enumerate(tp):
        bx.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, 1,
                                   facecolor=cmap(norm(v)), edgecolor="none"))
        bx.text(j, 0, f"{v:.1f}", ha="center", va="center", fontsize=6.4,
                color="white" if abs(v) > 7 else "black", zorder=3)
    bx.set_xlim(-0.5, NP - 0.5); bx.set_ylim(0.5, -0.5)
    bx.set_yticks([0])
    bx.set_yticklabels(["Mean $T_{\\mathrm{peak}}$ (\u00b0C)"], fontsize=7.5)
    bx.set_xticks(range(NP)); bx.set_xticklabels(PCTS * len(PERIODS), fontsize=7)
    for k in SEP:
        bx.axvline(k, color="white", lw=2.4, zorder=4)
    bx.tick_params(length=0); bx.spines[:].set_visible(False)
    bx.set_title("(b) Mean peak temperature", fontsize=8.5, loc="left", pad=4)
    season_labels(bx, 1.05)

    fig.tight_layout(); fig.subplots_adjust(hspace=0.34, bottom=0.17)
    fig.savefig("Fig_peak_matrix.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    coverage_matrix(); coverage_bars(); peak_matrix(); print("done")
