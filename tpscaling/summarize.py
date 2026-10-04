"""
Stage 2 - per-bin summary statistics, and stage 3 - log transform.

Stage 2 reduces each binned file (one column per temperature bin, ragged rows)
to one row per bin, holding the sum, mean, median, wet-day count and the P75,
P90, P95 and P99 values within that bin.

Stage 3 adds natural-log columns for the precipitation summaries only. Scaling
is fitted to ln(P) against temperature, so this is where ln(P) is created.
Temperature summaries are left untouched; the scaling fit uses their ``Mean``
column directly.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import PERCENTILES

#: Columns that stage 3 log-transforms, in the precipitation summaries.
LOG_COLUMNS = ("Median",) + tuple(f"{p}th Percentile" for p in
                                  sorted(PERCENTILES, reverse=True))


def summarize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce one binned file to one row per temperature bin."""
    rows = []
    for column in df.columns:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        row = {
            "Interval": column,
            "Sum": values.sum(),
            "Mean": values.mean(),
            "Median": values.median(),
            "Frequency": int(values.count()),
        }
        for p in sorted(PERCENTILES, reverse=True):
            row[f"{p}th Percentile"] = values.quantile(p / 100.0)
        rows.append(row)
    return pd.DataFrame(rows)


def run_summary(input_dir, output_dir, verbose: bool = True) -> int:
    """Stage 2 over a directory of binned files."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".csv"))
    for name in files:
        df = pd.read_csv(input_dir / name)
        out = summarize_frame(df)
        out.to_csv(output_dir / name.replace(".csv", "_summary.csv"),
                   index=False)
    if verbose:
        print(f"stage 2 complete: {len(files)} files summarized -> {output_dir}")
    return len(files)


def run_log_transform(summary_dir, columns=LOG_COLUMNS,
                      verbose: bool = True) -> int:
    """Stage 3 - add ``<column>_log`` columns to the precipitation summaries.

    Idempotent: re-running overwrites the log columns rather than appending
    duplicates. Zero and negative values become NaN, since ln is undefined
    there and an empty bin should not be read as ln(P) = 0.
    """
    summary_dir = Path(summary_dir)
    files = sorted(f for f in os.listdir(summary_dir) if f.endswith(".csv"))
    for name in files:
        path = summary_dir / name
        df = pd.read_csv(path)
        for column in columns:
            if column not in df.columns:
                continue
            values = pd.to_numeric(df[column], errors="coerce")
            df[f"{column}_log"] = np.where(values > 0, np.log(values), np.nan)
        df.to_csv(path, index=False)
    if verbose:
        print(f"stage 3 complete: log columns written for {len(files)} files")
    return len(files)
