"""
Atmospheric predictors: event matching and temperature regression.

Three steps, run in order after the binning stages.

**match** - For every temperature bin, the binned precipitation column holds
one value per wet day in that bin. The P75, P90, P95 and P99 values of that bin
each correspond to a particular day; this step finds the day whose
precipitation is closest to each percentile value and reads every predictor on
that same day. The output is therefore the atmospheric state accompanying the
extreme events that define each percentile, not a bin-mean of the predictor.

Ties and near-ties are resolved by taking the first closest match. Where a bin
holds few wet days the matched day may sit some distance from the nominal
percentile, so bins are required to be non-empty but no closeness tolerance is
imposed.

**regress** - Each matched predictor series is regressed by OLS on the mean
temperature of the corresponding bins, giving the rate at which that predictor
changes with temperature in that grid cell, per percentile.

**cluster** - Predictor slopes are grouped by the scaling regime of the same
grid cell, giving the mean predictor response within each regime.

The five predictors used in the published analysis are RH500, TCWV, VIMFC,
CAPE and OMEGA500.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from . import PERCENTILES

#: Predictors used in the published analysis.
PREDICTORS = ("RH500", "TCWV", "VIMFC", "CAPE", "OMEGA500")

#: Percentile column names, highest first.
PERCENTILE_COLUMNS = tuple(f"{p}th Percentile"
                           for p in sorted(PERCENTILES, reverse=True))


def _cell_key(path: Path) -> str:
    """``column_12_precipitation_result_summary.csv`` -> ``column_12``."""
    return "_".join(path.stem.split("_")[:2])


def _index(directory) -> dict:
    return {_cell_key(p): p for p in Path(directory).glob("*.csv")}


# ----------------------------------------------------------------- 1. MATCH
def match_one_cell(summary: pd.DataFrame, binned: pd.DataFrame,
                   predictor_frames: dict,
                   percentiles=PERCENTILE_COLUMNS) -> dict:
    """Match predictor values to the days defining each precipitation
    percentile, for one grid cell.

    Returns ``{"precipitation": DataFrame, <predictor>: DataFrame, ...}``, each
    with one row per temperature bin and one column per percentile.
    """
    if "Interval" not in summary.columns:
        raise ValueError("summary frame has no 'Interval' column")

    positions = []
    for _, row in summary.iterrows():
        interval = row["Interval"]
        if interval not in binned.columns:
            raise ValueError(f"interval {interval!r} missing from binned data")
        column = pd.to_numeric(binned[interval], errors="coerce")
        row_positions = {}
        for percentile in percentiles:
            target = row[percentile]
            if pd.isna(target) or column.notna().sum() == 0:
                row_positions[percentile] = None
            else:
                row_positions[percentile] = int((column - target).abs().idxmin())
        positions.append(row_positions)

    def gather(frame: pd.DataFrame) -> pd.DataFrame:
        records = []
        for i, (_, row) in enumerate(summary.iterrows()):
            interval = row["Interval"]
            column = pd.to_numeric(frame[interval], errors="coerce")
            record = {"Interval": interval}
            for percentile in percentiles:
                pos = positions[i][percentile]
                record[percentile] = (np.nan if pos is None
                                      else float(column.iloc[pos]))
            records.append(record)
        return pd.DataFrame(records)

    out = {"precipitation": gather(binned)}
    for name, frame in predictor_frames.items():
        out[name] = gather(frame)
    return out


def run_match(summary_dir, precipitation_bin_dir, predictor_bin_dirs,
              output_dir, percentiles=PERCENTILE_COLUMNS,
              verbose: bool = True) -> int:
    """Match predictors to percentile-defining days for every grid cell.

    ``predictor_bin_dirs`` maps predictor name to its binned directory.
    """
    summaries = _index(summary_dir)
    precipitation = _index(precipitation_bin_dir)
    predictors = {name: _index(d) for name, d in predictor_bin_dirs.items()}
    output_dir = Path(output_dir)

    done = 0
    for key, summary_path in sorted(summaries.items()):
        precip_path = precipitation.get(key)
        if precip_path is None:
            continue
        frames = {}
        missing = False
        for name, index in predictors.items():
            path = index.get(key)
            if path is None:
                missing = True
                break
            frames[name] = pd.read_csv(path)
        if missing:
            continue

        matched = match_one_cell(pd.read_csv(summary_path),
                                 pd.read_csv(precip_path), frames, percentiles)
        for name, frame in matched.items():
            folder = output_dir / (f"{name}_bin" if name != "precipitation"
                                   else "prec_bin")
            folder.mkdir(parents=True, exist_ok=True)
            frame.to_csv(folder / f"{key}_{name}_matched.csv", index=False)
        done += 1

    if verbose:
        print(f"matched predictors for {done} grid cells -> {output_dir}")
    return done


# -------------------------------------------------------------- 2. REGRESS
def regress_one(temperature: pd.Series, values: pd.Series) -> dict | None:
    """OLS of a matched predictor series on mean bin temperature."""
    frame = pd.DataFrame({"Mean": temperature, "y": values}).dropna()
    if len(frame) < 3:
        return None
    correlation, correlation_p = stats.pearsonr(frame["Mean"], frame["y"])
    model = sm.OLS(frame["y"], sm.add_constant(frame["Mean"])).fit()
    confidence = model.conf_int().loc["Mean"]
    return {
        "correlation_coef": float(correlation),
        "correlation_p": float(correlation_p),
        "slope": float(model.params["Mean"]),
        "intercept": float(model.params["const"]),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "regression_p": float(model.pvalues["Mean"]),
        "std_error": float(model.bse["Mean"]),
        "ci_low": float(confidence[0]),
        "ci_high": float(confidence[1]),
        "n_obs": int(model.nobs),
        "significant_0.05": bool(model.pvalues["Mean"] < 0.05),
    }


def run_regression(temperature_summary_dir, matched_dir, output_dir,
                   predictors=PREDICTORS, percentiles=PERCENTILE_COLUMNS,
                   verbose: bool = True) -> dict:
    """Regress every matched predictor on temperature, for every grid cell."""
    temperature_index = _index(temperature_summary_dir)
    matched_dir, output_dir = Path(matched_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {}
    for name in ("precipitation",) + tuple(predictors):
        folder = matched_dir / ("prec_bin" if name == "precipitation"
                                else f"{name}_bin")
        if not folder.is_dir():
            continue
        index = _index(folder)
        rows = {p: [] for p in percentiles}
        for key, path in sorted(index.items()):
            temperature_path = temperature_index.get(key)
            if temperature_path is None:
                continue
            temperature = pd.read_csv(temperature_path)["Mean"]
            frame = pd.read_csv(path)
            for percentile in percentiles:
                if percentile not in frame.columns:
                    continue
                result = regress_one(temperature, frame[percentile])
                if result is None:
                    continue
                rows[percentile].append({"file_id": key, "variable": name,
                                         "percentile": percentile, **result})

        for percentile, records in rows.items():
            if not records:
                continue
            table = pd.DataFrame(records)
            table["_n"] = table["file_id"].str.extract(r"(\d+)").astype(int)
            table = table.sort_values("_n").drop(columns="_n")
            label = percentile.replace(" ", "_")
            path = output_dir / f"{name}_{label}.csv"
            table.to_csv(path, index=False)
            tables[(name, percentile)] = table
            if verbose:
                print(f"  {name} {percentile}: {len(table)} cells")
    return tables


# --------------------------------------------------------------- 3. CLUSTER
def run_cluster(regression_dir, scaling_dir, output_file,
                predictors=PREDICTORS, percentiles=PERCENTILES,
                verbose: bool = True) -> pd.DataFrame:
    """Group predictor slopes by the scaling regime of the same grid cell.

    The regime label is read from the stage 4 scaling tables rather than from a
    column added by hand to the regression output, so the two can never drift
    apart.
    """
    regression_dir, scaling_dir = Path(regression_dir), Path(scaling_dir)
    from .scaling import REGIME_LABELS, normalize_regime
    order = list(REGIME_LABELS)

    records = []
    for percentile in percentiles:
        scaling_path = scaling_dir / f"P{percentile}.csv"
        if not scaling_path.exists():
            continue
        scaling = pd.read_csv(scaling_path)[["prefix", "scaling"]]
        scaling = scaling.rename(columns={"prefix": "file_id"})
        scaling["scaling"] = scaling["scaling"].map(normalize_regime)

        for predictor in predictors:
            path = regression_dir / f"{predictor}_{percentile}th_Percentile.csv"
            if not path.exists():
                if verbose:
                    print(f"  missing {path.name}")
                continue
            regression = pd.read_csv(path)[["file_id", "slope"]]
            merged = regression.merge(scaling, on="file_id", how="inner")
            if merged.empty:
                continue
            merged["scaling"] = pd.Categorical(merged["scaling"],
                                               categories=order, ordered=True)
            grouped = (merged.groupby("scaling", observed=False)["slope"]
                       .agg(["mean", "std", "count"]).reset_index())
            grouped.columns = ["scaling_regime", "mean_slope", "std_slope",
                               "n_cells"]
            grouped["predictor"] = predictor
            grouped["percentile"] = percentile
            records.append(grouped)

    if not records:
        return pd.DataFrame()
    combined = pd.concat(records, ignore_index=True)
    combined = combined[["predictor", "percentile", "scaling_regime",
                         "mean_slope", "std_slope", "n_cells"]]
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_file, index=False)
    if verbose:
        print(f"wrote {len(combined)} rows -> {output_file}")
    return combined
