"""
Modulation of scaling by atmospheric predictors, via quantile-regression
interaction models and the change in pseudo-R-squared.

For each grid cell and quantile tau, a baseline quantile regression of ln(P) on
temperature is fitted, then one model per predictor adding the predictor and
its interaction with temperature:

    baseline     ln(P) ~ T
    interaction  ln(P) ~ T + X + T:X

dR2 is the difference in Koenker-Machado pseudo-R-squared between the two.

**dR2 is attributable to the predictor and its interaction jointly, not to the
interaction alone**, because the interaction model adds two terms. It measures
how much of the precipitation-temperature relationship a predictor explains
beyond temperature by itself.

Standardisation
---------------
Temperature and every predictor are standardised to zero mean and unit variance
within each grid cell before fitting, so that interaction coefficients are
comparable across predictors with different physical units.

A consequence that matters downstream: the baseline temperature slope is then
**per standard deviation of temperature, not per degree C**. To convert, divide
by the cell's temperature standard deviation, which is written to the bivariate
output as ``temp_sd``:

    slope_per_degC = temp_slope / temp_sd

:func:`add_scaling_rate` does this and appends ``temp_slope_percent_per_degC``
and a ``scaling`` regime label, so that regimes derived here are directly
comparable with those from the percentile-binning pipeline. Never classify
regimes from ``temp_slope * 100`` directly: daily temperature standard
deviations over the plateau are of order 5-10 degC, so that quantity overstates
the scaling rate by roughly the same factor and pushes nearly every grid cell
into the super-C-C category.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .predictors import PREDICTORS
from .quantreg import QUANTILES, WET_DAY_THRESHOLD

MIN_OBSERVATIONS = 100


def _numerical_sort(filename: str) -> int:
    match = re.search(r"column_(\d+)\.csv", filename)
    return int(match.group(1)) if match else 0


def _fit(endog, exog, tau: float, max_iter: int = 5000, p_tol: float = 1e-6):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            return sm.QuantReg(endog, exog).fit(q=tau, max_iter=max_iter,
                                                p_tol=p_tol)
        except Exception:                                      # noqa: BLE001
            return None


def _standardise(frame: pd.DataFrame, columns) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            continue
        sd = out[column].std()
        out[f"{column}_std"] = ((out[column] - out[column].mean()) / sd
                                if sd > 0 else 0.0)
    return out


def analyse_cell(frame: pd.DataFrame, predictors=PREDICTORS,
                 quantiles=QUANTILES, threshold: float = WET_DAY_THRESHOLD,
                 min_observations: int = MIN_OBSERVATIONS) -> dict:
    """Run the baseline and interaction models for one grid cell.

    ``frame`` must hold ``precip``, ``temp`` and one column per predictor.
    """
    frame = frame.dropna()
    frame = frame[frame["precip"] >= threshold]
    if len(frame) < min_observations:
        return {}

    frame = _standardise(frame, ["temp"] + list(predictors))
    frame = frame.copy()
    frame["log_precip"] = np.log(frame["precip"])

    usable = [p for p in predictors
              if f"{p}_std" in frame.columns and frame[f"{p}_std"].nunique() > 1]
    if not usable:
        return {}

    temperature_sd = float(frame["temp"].std())
    results = {}

    for tau in quantiles:
        baseline = _fit(frame["log_precip"],
                        sm.add_constant(frame[["temp_std"]]), tau)
        if baseline is None or np.isnan(baseline.params).any():
            continue
        baseline_r2 = float(getattr(baseline, "prsquared", np.nan))

        interactions = {}
        for predictor in usable:
            term = f"temp_std_x_{predictor}_std"
            design = frame[["temp_std", f"{predictor}_std"]].copy()
            design[term] = frame["temp_std"] * frame[f"{predictor}_std"]
            model = _fit(frame["log_precip"], sm.add_constant(design), tau)
            if (model is None or np.isnan(model.params).any()
                    or term not in model.params):
                continue
            interaction_r2 = float(getattr(model, "prsquared", np.nan))
            interactions[predictor] = {
                "interaction_coefficient": float(model.params[term]),
                "interaction_pvalue": float(model.pvalues[term]),
                "predictor_coefficient": float(model.params[f"{predictor}_std"]),
                "baseline_temp_coef": float(baseline.params["temp_std"]),
                "interaction_temp_coef": float(model.params["temp_std"]),
                "baseline_r2": baseline_r2,
                "interaction_r2": interaction_r2,
                "delta_r2": interaction_r2 - baseline_r2,
            }

        results[tau] = {
            "temp_slope": float(baseline.params["temp_std"]),
            "temp_pvalue": float(baseline.pvalues["temp_std"]),
            "const_coef": float(baseline.params["const"]),
            "r2": baseline_r2,
            "n_observations": int(len(frame)),
            "temp_mean": float(frame["temp"].mean()),
            "temp_sd": temperature_sd,
            "standardized": True,
            "interactions": interactions,
        }
    return results


def add_scaling_rate(bivariate: pd.DataFrame) -> pd.DataFrame:
    """Convert standardised slopes to %/degC and attach a regime label."""
    from .scaling import categorize_slope

    out = bivariate.copy()
    out["temp_slope_percent_per_degC"] = (
        out["temp_slope"] / out["temp_sd"].replace(0, np.nan) * 100.0)
    out["scaling"] = out["temp_slope_percent_per_degC"].map(categorize_slope)
    return out


def run(precipitation_dir, temperature_dir, predictor_dirs, output_dir,
        quantiles=QUANTILES, threshold: float = WET_DAY_THRESHOLD,
        processes: int | None = None, verbose: bool = True) -> dict:
    """Run the modulation analysis over every grid cell.

    Writes ``bivariate/tau_XX.csv`` and ``change_in_r2/tau_XX.csv`` under
    ``output_dir``. The bivariate tables carry both the standardised slope and
    the converted %/degC rate with its regime label.
    """
    precipitation_dir = Path(precipitation_dir)
    temperature_dir = Path(temperature_dir)
    predictor_dirs = {k: Path(v) for k, v in predictor_dirs.items()}
    output_dir = Path(output_dir)
    (output_dir / "bivariate").mkdir(parents=True, exist_ok=True)
    (output_dir / "change_in_r2").mkdir(parents=True, exist_ok=True)

    names = sorted((f for f in os.listdir(precipitation_dir)
                    if f.endswith(".csv")), key=_numerical_sort)

    bivariate_rows = {tau: [] for tau in quantiles}
    delta_rows = {tau: [] for tau in quantiles}

    for name in names:
        if not (temperature_dir / name).exists():
            continue
        columns = {
            "precip": pd.read_csv(precipitation_dir / name,
                                  header=None).to_numpy(float).ravel(),
            "temp": pd.read_csv(temperature_dir / name,
                                header=None).to_numpy(float).ravel(),
        }
        missing = False
        for predictor, folder in predictor_dirs.items():
            path = folder / name
            if not path.exists():
                missing = True
                break
            columns[predictor] = pd.read_csv(
                path, header=None).to_numpy(float).ravel()
        if missing:
            continue
        lengths = {k: v.size for k, v in columns.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"{name}: variables differ in length {lengths}")

        cell = Path(name).stem
        results = analyse_cell(pd.DataFrame(columns),
                               predictors=tuple(predictor_dirs),
                               quantiles=quantiles, threshold=threshold)
        for tau, record in results.items():
            interactions = record.pop("interactions")
            bivariate_rows[tau].append({"grid_cell": cell, **record})
            for predictor, metrics in interactions.items():
                delta_rows[tau].append({"grid_cell": cell,
                                        "predictor": predictor,
                                        "tau": tau, **metrics})

    tables = {}
    for tau in quantiles:
        label = f"tau_{int(tau * 100):02d}.csv"
        bivariate = pd.DataFrame(bivariate_rows[tau])
        if not bivariate.empty:
            bivariate = add_scaling_rate(bivariate)
        bivariate.to_csv(output_dir / "bivariate" / label, index=False)
        delta = pd.DataFrame(delta_rows[tau])
        delta.to_csv(output_dir / "change_in_r2" / label, index=False)
        tables[tau] = (bivariate, delta)
        if verbose:
            print(f"  tau={tau}: {len(bivariate)} cells, "
                  f"{len(delta)} predictor interactions")
    return tables
