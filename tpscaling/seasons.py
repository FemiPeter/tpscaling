"""
Seasonal subsetting of daily grid-cell series.

Each input CSV is a headerless daily series for one grid cell, covering the
same calendar days as a companion date file (one ISO date per line, no header).
Rows are assigned to DJF, MAM, JJA or SON and written to one subdirectory per
season, keeping the original file name so downstream stages can pair files by
name across variables.

Winter is calendar DJF: December belongs to the same season label as the
January and February that precede it, not the ones that follow. This matters
only if winters are ever treated as continuous blocks; here each day is scored
independently, so the convention has no effect on the scaling estimates.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

SEASONS = ("Winter", "Spring", "Summer", "Autumn")

SEASON_OF_MONTH = {12: "Winter", 1: "Winter", 2: "Winter",
                   3: "Spring", 4: "Spring", 5: "Spring",
                   6: "Summer", 7: "Summer", 8: "Summer",
                   9: "Autumn", 10: "Autumn", 11: "Autumn"}


def load_dates(date_file) -> pd.Series:
    """Read the companion date file as a datetime series."""
    dates = pd.read_csv(date_file, header=None, names=["Date"])
    return pd.to_datetime(dates["Date"])


def run(input_dir, date_file, output_dir=None, seasons=SEASONS,
        verbose: bool = True) -> int:
    """Split every CSV in ``input_dir`` into per-season subdirectories.

    ``output_dir`` defaults to ``input_dir``, which writes the season folders
    alongside the annual files, as in the original analysis.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir
    dates = load_dates(date_file)
    labels = dates.dt.month.map(SEASON_OF_MONTH)

    for season in seasons:
        (output_dir / season).mkdir(parents=True, exist_ok=True)

    date_name = Path(date_file).name
    files = sorted(f for f in os.listdir(input_dir)
                   if f.endswith(".csv") and f != date_name)

    for name in files:
        data = pd.read_csv(input_dir / name, header=None)
        # Files are normally one row per day. A year-by-day matrix is accepted
        # too and flattened row-major, which is the order the binning stage
        # also uses, so the two representations stay interchangeable.
        if len(data) != len(dates):
            if data.size == len(dates):
                data = pd.DataFrame(data.to_numpy().ravel())
            else:
                raise ValueError(
                    f"{name} holds {data.size} values but the date file has "
                    f"{len(dates)}; they must correspond one to one")
        for season in seasons:
            subset = data[labels.to_numpy() == season]
            subset.to_csv(output_dir / season / name, index=False, header=False)

    if verbose:
        print(f"split {len(files)} files into {', '.join(seasons)}")
    return len(files)
