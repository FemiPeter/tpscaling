"""
Stage 4 - scaling fit, structure detection and regime classification.

For each grid cell and each extreme percentile, ln(P) is regressed on mean bin
temperature. Before fitting, a LOESS curve locates the maximum of the
relationship. If that maximum is interior, the cell has a peak structure and
the fit is restricted to the rising side (up to the peak if the peak lies in
the warm half, from the peak onward if it lies in the cool half). If the
maximum sits at an endpoint, the relationship is monotonic and the full range
is fitted.

The slope is converted to a percentage change per degree C and classified into
four regimes:

    negative      < 0 %/degC
    sub-C-C       0 to 5 %/degC
    C-C           5 to 9 %/degC
    super-C-C     > 9 %/degC
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error, r2_score

from . import PERCENTILES

#: Minimum number of usable temperature bins for a fit to be attempted.
MIN_BINS = 10

#: LOESS smoothing fraction used to locate the peak.
LOESS_FRAC = 0.3

#: Significance level for the reported slope.
ALPHA = 0.05

#: Scaling-regime bounds, in %/degC. Edit here to change the classification.
REGIME_BOUNDS = ((0.0, "Negative Scaling"),
                 (5.0, "Sub C-C scaling"),
                 (9.0, "C-C scaling"),
                 (np.inf, "Super C-C scaling"))


#: Canonical regime labels, in the order used everywhere.
REGIME_LABELS = tuple(label for _, label in REGIME_BOUNDS)


def normalize_regime(label):
    """Map any spelling of a regime label onto the canonical one.

    Result files written at different times use ``C-C scaling`` or ``C-C like
    scaling``, and vary in capitalisation. Every merge and grouping in this
    package passes labels through here first, so a file written under either
    convention still joins correctly.
    """
    if not isinstance(label, str):
        return np.nan
    text = label.strip().lower().replace("\u2013", "-").replace("_", " ")
    if "super" in text:
        return "Super C-C scaling"
    if "sub" in text:
        return "Sub C-C scaling"
    if "negative" in text:
        return "Negative Scaling"
    if "c-c" in text:
        return "C-C scaling"
    return np.nan


def categorize_slope(slope_pct_per_c: float) -> str | float:
    """Map a scaling rate in %/degC onto a scaling regime."""
    if slope_pct_per_c is None or not np.isfinite(slope_pct_per_c):
        return np.nan
    if slope_pct_per_c < 0:
        return REGIME_BOUNDS[0][1]
    for bound, label in REGIME_BOUNDS[1:]:
        if slope_pct_per_c < bound:
            return label
    return REGIME_BOUNDS[-1][1]


def analyze_cell(temperature: np.ndarray, log_precipitation: np.ndarray,
                 loess_frac: float = LOESS_FRAC, min_bins: int = MIN_BINS,
                 alpha: float = ALPHA) -> dict:
    """Fit one grid cell at one percentile.

    ``temperature`` is the per-bin mean temperature and ``log_precipitation``
    the per-bin ln(P) of the chosen percentile. Both are one value per bin.
    """
    t = np.asarray(temperature, dtype=float)
    y = np.asarray(log_precipitation, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]

    empty = dict(relationship="insufficient_data", r2_value=np.nan,
                 peak_side=None, peak_temp=np.nan, slope=np.nan,
                 r_value=np.nan, p_value=np.nan, rmse=np.nan,
                 statistical_significance="FALSE", slope2=np.nan,
                 scaling=np.nan, n_bins=int(ok.sum()))
    if t.size < min_bins:
        return empty

    order = np.argsort(t)
    t, y = t[order], y[order]

    smoothed = sm.nonparametric.lowess(y, t, frac=loess_frac)
    smooth_t, smooth_y = smoothed[:, 0], smoothed[:, 1]
    peak_index = int(np.argmax(smooth_y))
    peak_temp = float(smooth_t[peak_index])
    is_peak = 0 < peak_index < len(smooth_y) - 1

    if is_peak:
        # A peak in the warm half means precipitation rises up to it; a peak in
        # the cool half means it falls away from it. Either way the segment
        # fitted is the one the scaling rate is meant to describe.
        if peak_temp > float(np.median(t)):
            mask, peak_side = t <= peak_temp, "right"
        else:
            mask, peak_side = t >= peak_temp, "left"
    else:
        mask, peak_side = np.ones_like(t, dtype=bool), None

    if mask.sum() < 3:
        return empty

    slope, intercept, r_value, p_value, _ = linregress(t[mask], y[mask])
    fitted = intercept + slope * t[mask]
    slope2 = slope * 100.0

    return dict(
        relationship="peak_structure" if is_peak else "monotonic",
        r2_value=float(r2_score(y[mask], fitted)),
        peak_side=peak_side,
        peak_temp=peak_temp if is_peak else np.nan,
        slope=float(slope),
        r_value=float(r_value),
        p_value=float(p_value),
        rmse=float(np.sqrt(mean_squared_error(y[mask], fitted))),
        statistical_significance="TRUE" if p_value < alpha else "FALSE",
        slope2=float(slope2),
        scaling=categorize_slope(slope2),
        n_bins=int(mask.sum()),
    )


def _numeric_suffix(prefix: str) -> int:
    m = re.search(r"(\d+)", prefix)
    return int(m.group(1)) if m else -1


def run(temperature_summary_dir, precipitation_summary_dir, output_dir,
        percentiles=PERCENTILES, loess_frac: float = LOESS_FRAC,
        min_bins: int = MIN_BINS, verbose: bool = True) -> dict:
    """Fit every grid cell at every percentile; one output CSV per percentile."""
    tdir = Path(temperature_summary_dir)
    pdir = Path(precipitation_summary_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_temperature_result_summary.csv"
    names = sorted(f for f in os.listdir(tdir) if f.endswith(suffix))

    tables = {}
    for percentile in percentiles:
        column = f"{percentile}th Percentile_log"
        rows = []
        for name in names:
            prefix = name[: -len(suffix)]
            pfile = pdir / f"{prefix}_precipitation_result_summary.csv"
            if not pfile.exists():
                continue
            tdf = pd.read_csv(tdir / name)
            pdf = pd.read_csv(pfile)
            if "Mean" not in tdf.columns or column not in pdf.columns:
                continue
            result = analyze_cell(tdf["Mean"].to_numpy(),
                                  pdf[column].to_numpy(),
                                  loess_frac=loess_frac, min_bins=min_bins)
            result["prefix"] = prefix
            result["percentile"] = f"{percentile}th Percentile_log"
            rows.append(result)

        table = pd.DataFrame(rows)
        if not table.empty:
            table["_n"] = table["prefix"].map(_numeric_suffix)
            table = (table.sort_values("_n").drop(columns="_n")
                     .reset_index(drop=True))
            ordered = ["prefix", "percentile", "relationship", "r2_value",
                       "peak_side", "peak_temp", "slope", "r_value", "p_value",
                       "rmse", "statistical_significance", "slope2", "scaling",
                       "n_bins"]
            table = table[[c for c in ordered if c in table.columns]]

        out = output_dir / f"P{percentile}.csv"
        table.to_csv(out, index=False)
        tables[percentile] = table
        if verbose and not table.empty:
            counts = table["relationship"].value_counts().to_dict()
            print(f"  P{percentile}: {len(table)} cells -> {out.name}  {counts}")

    if verbose:
        print(f"stage 4 complete: {len(tables)} percentile tables written")
    return tables
