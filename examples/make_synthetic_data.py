"""Generate a small synthetic dataset so the pipeline can be run end to end.

Creates 12 fake grid cells with 50 years x 365 days of paired daily
temperature and precipitation. Cells are given different scaling behaviours so
that all four regimes and both structure types appear in the output. The
values are invented and carry no scientific meaning; they exist only to
demonstrate that the code runs and to let a reader check the output format.
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(0)
root = Path(__file__).resolve().parent / "synthetic"
(root / "tmean").mkdir(parents=True, exist_ok=True)
(root / "prec").mkdir(parents=True, exist_ok=True)

NYEAR, NDAY = 50, 365
# (scaling rate in ln(P) per degC, peak temperature or None)
CELLS = [(0.115, None), (0.115, 12.0), (0.072, None), (0.072, 10.0),
         (0.072, 8.0), (0.030, None), (0.030, 9.0), (0.030, 6.0),
         (-0.035, None), (-0.035, 5.0), (0.072, 14.0), (0.115, None)]

for i, (slope, peak) in enumerate(CELLS, start=1):
    doy = np.tile(np.arange(NDAY), NYEAR)
    temp = (8 + 14 * np.sin(2 * np.pi * (doy - 100) / 365)
            + rng.normal(0, 3.0, NYEAR * NDAY))
    lnp = 0.4 + slope * (temp - temp.mean())
    if peak is not None:
        lnp -= (abs(slope) * 1.9 + 0.02) * np.clip(temp - peak, 0, None)
    prec = np.exp(lnp + rng.normal(0, 0.7, temp.size))
    prec[rng.random(temp.size) < 0.55] = 0.0          # dry days

    pd.DataFrame(temp.reshape(NYEAR, NDAY)).to_csv(
        root / "tmean" / f"column_{i}.csv", header=False, index=False)
    pd.DataFrame(prec.reshape(NYEAR, NDAY)).to_csv(
        root / "prec" / f"column_{i}.csv", header=False, index=False)

    # Five atmospheric predictors, loosely tied to temperature so the
    # downstream matching and regression stages have something to find.
    predictors = {
        "RH500":    55 + 0.9 * (temp - temp.mean()) + rng.normal(0, 6, temp.size),
        "TCWV":     8 + 0.45 * (temp - temp.mean()) + rng.normal(0, 1.5, temp.size),
        "VIMFC":    rng.normal(0, 2e-5, temp.size) + 4e-6 * (temp - temp.mean()),
        "CAPE":     np.clip(80 + 6 * (temp - temp.mean())
                            + rng.normal(0, 40, temp.size), 0, None),
        "OMEGA500": rng.normal(0, 0.08, temp.size) - 0.004 * (temp - temp.mean()),
    }
    for name, values in predictors.items():
        (root / name).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(values.reshape(NYEAR, NDAY)).to_csv(
            root / name / f"column_{i}.csv", header=False, index=False)

dates = pd.date_range("1970-01-01", periods=NYEAR * NDAY, freq="D")
pd.Series(dates.strftime("%Y-%m-%d")).to_csv(
    root / "daily_dates.csv", header=False, index=False)

print(f"wrote {len(CELLS)} synthetic grid cells to {root}")
