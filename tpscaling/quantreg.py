"""
Bivariate quantile regression of ln(P) on temperature.

This is the independent estimate of the scaling rate used to check the
percentile-binning result. For each grid cell and each quantile tau, a quantile
regression of ln(P) on daily mean temperature is fitted to all wet days, and
the slope is reported as a percentage change per degree C. A k-fold
cross-validation gives the spread of the slope across subsets of the record.

Folds are contiguous blocks of the time series, not random draws: ``KFold`` is
used without shuffling. The spread across folds therefore reflects variation
between periods of the record, which is the quantity of interest, rather than
the sampling noise of random subsets.

Unlike the binning estimate, this stage fits the full temperature range and
does not detect peak structures, so the two agree most closely where the
relationship is monotonic.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold

#: Quantiles fitted, matching the P75-P99 extreme percentiles.
QUANTILES = (0.75, 0.90, 0.95, 0.99)

#: Wet-day threshold (mm), matching stage 1.
WET_DAY_THRESHOLD = 0.1

#: Minimum wet days for a cell to be fitted.
MIN_OBSERVATIONS = 100

#: Minimum wet days in a fold for that fold to contribute.
MIN_FOLD_OBSERVATIONS = 20

N_FOLDS = 5


def _numerical_sort(filename: str) -> int:
    match = re.search(r"column_(\d+)\.csv", filename)
    return int(match.group(1)) if match else 0


def _fit(endog, exog, tau: float, max_iter: int = 5000, p_tol: float = 1e-6):
    """Fit one quantile regression, returning None if it fails to converge."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            return sm.QuantReg(endog, exog).fit(q=tau, max_iter=max_iter,
                                                p_tol=p_tol)
        except Exception as exc:                              # noqa: BLE001
            print(f"  fit failed at tau={tau}: {exc}")
            return None


def fit_cell(temperature: np.ndarray, precipitation: np.ndarray,
             quantiles=QUANTILES, n_folds: int = N_FOLDS,
             threshold: float = WET_DAY_THRESHOLD) -> dict:
    """Fit all quantiles for one grid cell. Returns ``{tau: {...}}``."""
    df = pd.DataFrame({"temp": np.asarray(temperature, float).ravel(),
                       "precip": np.asarray(precipitation, float).ravel()})
    df = df.dropna()
    df = df[df["precip"] >= threshold]
    if len(df) < MIN_OBSERVATIONS or df["temp"].nunique() == 1:
        return {}
    df["log_precip"] = np.log(df["precip"])

    out = {}
    for tau in quantiles:
        full = _fit(df["log_precip"], sm.add_constant(df["temp"]), tau)
        if full is None or np.isnan(full.params).any():
            continue
        full_slope = float(full.params["temp"])

        fold_slopes = []
        for train_idx, _ in KFold(n_splits=n_folds).split(df):
            train = df.iloc[train_idx]
            if (len(train) < MIN_FOLD_OBSERVATIONS
                    or train["temp"].nunique() == 1):
                fold_slopes.append(np.nan)
                continue
            fold = _fit(train["log_precip"], sm.add_constant(train["temp"]),
                        tau)
            fold_slopes.append(float(fold.params["temp"])
                               if fold is not None and "temp" in fold.params
                               else np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_cv = float(np.nanmean(fold_slopes))
            std_cv = float(np.nanstd(fold_slopes, ddof=1))

        out[tau] = {
            "full_slope": full_slope,
            "mean_cv_slope": mean_cv,
            "std_cv_slope": std_cv,
            "full_slope_percent": full_slope * 100.0,
            "mean_cv_slope_percent": mean_cv * 100.0,
            "std_cv_slope_percent": std_cv * 100.0,
            "n_obs": int(len(df)),
            **{f"fold_{i + 1}": s for i, s in enumerate(fold_slopes)},
        }
    return out


def run(temperature_dir, precipitation_dir, output_dir,
        quantiles=QUANTILES, n_folds: int = N_FOLDS,
        threshold: float = WET_DAY_THRESHOLD, verbose: bool = True) -> dict:
    """Fit every grid cell; writes one ``tau_XX.csv`` per quantile."""
    tdir, pdir = Path(temperature_dir), Path(precipitation_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    precip_files = sorted((f for f in os.listdir(pdir) if f.endswith(".csv")),
                          key=_numerical_sort)
    temp_files = sorted((f for f in os.listdir(tdir) if f.endswith(".csv")),
                        key=_numerical_sort)
    if precip_files != temp_files:
        missing = set(precip_files) ^ set(temp_files)
        raise ValueError(
            f"precipitation and temperature file lists differ; "
            f"{len(missing)} file(s) unmatched, e.g. {sorted(missing)[:3]}")

    rows = {tau: [] for tau in quantiles}
    for name in precip_files:
        precipitation = pd.read_csv(pdir / name, header=None).to_numpy(float)
        temperature = pd.read_csv(tdir / name, header=None).to_numpy(float)
        results = fit_cell(temperature, precipitation, quantiles, n_folds,
                           threshold)
        for tau, record in results.items():
            rows[tau].append({"grid_cell": Path(name).stem, **record})

    tables = {}
    for tau in quantiles:
        table = pd.DataFrame(rows[tau])
        # Written once at the end rather than appended per cell, so re-running
        # replaces the output instead of duplicating rows into it.
        path = output_dir / f"tau_{int(tau * 100):02d}.csv"
        table.to_csv(path, index=False)
        tables[tau] = table
        if verbose:
            print(f"  tau={tau}: {len(table)} cells -> {path.name}")
    return tables
