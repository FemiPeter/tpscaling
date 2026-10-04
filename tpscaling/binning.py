"""
Stage 1 - temperature-percentile binning.

For every grid cell, paired daily precipitation and temperature series are
pooled, restricted to wet days, and sorted into temperature bins defined by
percentiles of the wet-day temperature distribution for that cell. Bin edges
are therefore cell-specific: bin k is the k-th coldest slice of that cell's wet
days, not a fixed temperature range.

Input
-----
Two directories holding one CSV per grid cell, with matching file names
(for example ``column_1.csv`` in both). Each file is a headerless numeric
matrix of daily values; the shape does not matter because values are flattened.

Output
------
Two directories of ``<cell>_precipitation_result.csv`` and
``<cell>_temperature_result.csv``. Each has one column per temperature bin,
holding the wet-day values that fall in that bin. Columns are ragged, so
shorter columns are padded with blanks.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

#: Wet-day threshold (mm). Days below this are excluded entirely.
WET_DAY_THRESHOLD = 0.1

#: Percentile edges of the wet-day temperature distribution, giving 20 bins.
BIN_EDGES_PERCENT = tuple(range(0, 101, 5))


def bin_edges(temperature: np.ndarray,
              edges_percent: tuple = BIN_EDGES_PERCENT) -> np.ndarray:
    """Percentile edges of the wet-day temperature distribution."""
    return np.percentile(temperature, list(edges_percent))


def interval_labels(edges: np.ndarray) -> list[str]:
    """Column names of the form ``temp<lo>-<hi>``, matching the original run."""
    return [f"temp{edges[i]}-{edges[i + 1]}" for i in range(len(edges) - 1)]


def bin_one_cell(temperature: np.ndarray,
                 precipitation: np.ndarray,
                 threshold: float = WET_DAY_THRESHOLD,
                 edges_percent: tuple = BIN_EDGES_PERCENT):
    """Bin one grid cell's paired series.

    Returns ``(temperature_frame, precipitation_frame, labels)``. Bins are
    closed on the right and the lowest bin is closed on both sides, so a value
    sitting exactly on an interior edge falls in the lower bin.
    """
    temperature = np.asarray(temperature, dtype=float).ravel()
    precipitation = np.asarray(precipitation, dtype=float).ravel()
    if temperature.shape != precipitation.shape:
        raise ValueError(
            f"shape mismatch: temperature {temperature.shape} vs "
            f"precipitation {precipitation.shape}")

    wet = np.isfinite(temperature) & np.isfinite(precipitation)
    wet &= precipitation >= threshold
    t, p = temperature[wet], precipitation[wet]
    if t.size == 0:
        return None, None, []

    edges = bin_edges(t, edges_percent)
    labels = interval_labels(edges)

    # Duplicate edges arise when the temperature distribution is very flat;
    # dropping them loses a bin rather than failing outright.
    uniq = np.unique(edges)
    if uniq.size < edges.size:
        idx = pd.cut(t, bins=uniq, labels=False, include_lowest=True)
        keep = [i for i in range(len(edges) - 1) if edges[i] != edges[i + 1]]
        labels = [labels[i] for i in keep]
    else:
        idx = pd.cut(t, bins=edges, labels=False, include_lowest=True)

    n_bins = len(labels)
    t_cols = [t[idx == k] for k in range(n_bins)]
    p_cols = [p[idx == k] for k in range(n_bins)]

    t_df = pd.DataFrame({lab: pd.Series(col)
                         for lab, col in zip(labels, t_cols)})
    p_df = pd.DataFrame({lab: pd.Series(col)
                         for lab, col in zip(labels, p_cols)})
    return t_df, p_df, labels


def run(temperature_dir, precipitation_dir, out_temperature_dir,
        out_precipitation_dir, threshold: float = WET_DAY_THRESHOLD,
        edges_percent: tuple = BIN_EDGES_PERCENT, verbose: bool = True) -> int:
    """Bin every grid cell found in both input directories."""
    temperature_dir = Path(temperature_dir)
    precipitation_dir = Path(precipitation_dir)
    out_t = Path(out_temperature_dir)
    out_p = Path(out_precipitation_dir)
    out_t.mkdir(parents=True, exist_ok=True)
    out_p.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in os.listdir(precipitation_dir)
                   if f.endswith(".csv"))
    done = 0
    for name in files:
        tfile = temperature_dir / name
        if not tfile.exists():
            if verbose:
                print(f"  skip {name}: no matching temperature file")
            continue

        temperature = pd.read_csv(tfile, header=None).to_numpy(dtype=float)
        precipitation = pd.read_csv(precipitation_dir / name,
                                    header=None).to_numpy(dtype=float)
        t_df, p_df, labels = bin_one_cell(temperature, precipitation,
                                          threshold, edges_percent)
        if t_df is None:
            if verbose:
                print(f"  skip {name}: no wet days")
            continue

        stem = Path(name).stem
        t_df.to_csv(out_t / f"{stem}_temperature_result.csv", index=False)
        p_df.to_csv(out_p / f"{stem}_precipitation_result.csv", index=False)
        done += 1
        if verbose and done % 200 == 0:
            print(f"  binned {done}/{len(files)}")

    if verbose:
        print(f"stage 1 complete: {done} grid cells binned")
    return done


def run_multivariable(variable_dirs, output_dirs,
                      threshold: float = WET_DAY_THRESHOLD,
                      edges_percent: tuple = BIN_EDGES_PERCENT,
                      verbose: bool = True) -> int:
    """Bin precipitation, temperature and atmospheric predictors together.

    Bins are defined by the wet-day temperature distribution, exactly as in
    :func:`run`, and every variable is sorted into those same bins using the
    same day indices. Each variable therefore keeps its day-by-day
    correspondence with precipitation, which is what lets
    :mod:`tpscaling.predictors` later read a predictor on the specific day that
    defines a given precipitation percentile.

    ``variable_dirs`` and ``output_dirs`` map variable name to directory, and
    must both contain the keys ``"precipitation"`` and ``"temperature"``.
    """
    required = {"precipitation", "temperature"}
    missing = required - set(variable_dirs)
    if missing:
        raise ValueError(f"variable_dirs is missing {sorted(missing)}")
    if set(variable_dirs) != set(output_dirs):
        raise ValueError("variable_dirs and output_dirs must have the same keys")

    variable_dirs = {k: Path(v) for k, v in variable_dirs.items()}
    output_dirs = {k: Path(v) for k, v in output_dirs.items()}
    for folder in output_dirs.values():
        folder.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in os.listdir(variable_dirs["precipitation"])
                   if f.endswith(".csv"))
    done = 0
    for name in files:
        paths = {var: folder / name for var, folder in variable_dirs.items()}
        absent = [var for var, path in paths.items() if not path.exists()]
        if absent:
            if verbose:
                print(f"  skip {name}: no data for {', '.join(absent)}")
            continue

        data = {var: pd.read_csv(path, header=None).to_numpy(dtype=float).ravel()
                for var, path in paths.items()}
        shapes = {var: arr.shape for var, arr in data.items()}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"{name}: variables have different lengths {shapes}")

        precipitation = data["precipitation"]
        temperature = data["temperature"]
        wet = (np.isfinite(precipitation) & np.isfinite(temperature)
               & (precipitation >= threshold))
        if wet.sum() == 0:
            continue

        edges = bin_edges(temperature[wet], edges_percent)
        labels = interval_labels(edges)
        uniq = np.unique(edges)
        if uniq.size < edges.size:
            idx = pd.cut(temperature[wet], bins=uniq, labels=False,
                         include_lowest=True)
            keep = [i for i in range(len(edges) - 1) if edges[i] != edges[i + 1]]
            labels = [labels[i] for i in keep]
        else:
            idx = pd.cut(temperature[wet], bins=edges, labels=False,
                         include_lowest=True)

        stem = Path(name).stem
        for var, values in data.items():
            masked = values[wet]
            frame = pd.DataFrame(
                {label: pd.Series(masked[idx == k])
                 for k, label in enumerate(labels)})
            frame.to_csv(output_dirs[var] / f"{stem}_{var}_result.csv",
                         index=False)
        done += 1

    if verbose:
        print(f"multivariable binning complete: {done} grid cells, "
              f"{len(variable_dirs)} variables")
    return done
