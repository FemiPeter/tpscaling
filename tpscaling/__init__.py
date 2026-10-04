"""
tpscaling - extreme precipitation-temperature scaling over the Tibetan Plateau.

Preparation
    era5-download   retrieve an ERA5 variable over the plateau
    era5-extract    sample a NetCDF field at the plateau grid cells
    vimfc           VIMFC from ERA5 moisture flux components
    split-seasons   subset daily series into DJF, MAM, JJA, SON

Core scaling pipeline (`tpscaling run-all`)
    1. bin            temperature-percentile binning of paired daily series
    2. summarize      per-bin statistics (sum, mean, median, count, percentiles)
    3. logtransform   natural log of the precipitation percentile columns
    4. scale          LOESS peak detection and linear scaling fit per grid cell,
                      with scaling-regime classification

Modulation analysis
    bin-predictors      bin precipitation, temperature and five predictors together
    match-predictors    read predictors on the days defining each percentile
    regress-predictors  regress matched predictors on bin temperature
    cluster             group predictor slopes by scaling regime
    modulation          quantile-regression interaction models and dR2

Validation
    quantreg        bivariate quantile regression, as an independent estimate

Each stage reads the output directory of the previous one, so stages can be run
individually or chained.
"""

__version__ = "1.0.0"

PERCENTILES = (75, 90, 95, 99)

__all__ = ["PERCENTILES", "__version__"]
