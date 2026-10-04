# tpscaling

Analysis code for extreme precipitation–temperature (P–T) scaling over the
Tibetan Plateau.

This repository contains the pipeline used to classify daily extreme
precipitation scaling into four regimes (negative, sub-C–C, C–C, super-C–C) and
two structure types (monotonic, peak structure) at annual and seasonal time
scales, for the P75, P90, P95 and P99 extreme percentiles.

## What the pipeline does

Scaling is estimated by temperature-percentile binning. For each grid cell,
paired daily precipitation and temperature series are pooled over wet days,
sorted into 20 bins defined by percentiles of that cell's own wet-day
temperature distribution, and reduced to one summary row per bin. The natural
logarithm of each precipitation percentile is then regressed on mean bin
temperature, and the resulting slope is expressed as a percentage change per
degree Celsius.

Because bin edges are percentile-based, they are specific to each grid cell:
bin *k* is the *k*-th coldest slice of that cell's wet days, not a fixed
temperature range. Bins are therefore comparable across cells by rank, not by
absolute temperature.

Before fitting, a LOESS curve locates the maximum of the ln(P)–temperature
relationship. An interior maximum marks a **peak structure**, and the
regression is restricted to the rising side of the peak; a maximum at either
end of the temperature range marks a **monotonic** relationship, and the full
range is fitted.

## Stages

**Preparation**

| Command | Purpose |
|---|---|
| `era5-download` | retrieve an ERA5 variable over the plateau, one NetCDF per year |
| `era5-extract` | sample a NetCDF field at the 4,019 plateau grid cells, writing one CSV per cell |
| `vimfc` | vertically integrated moisture flux convergence from ERA5 `viwve` and `viwvn` |
| `split-seasons` | subset daily series into DJF, MAM, JJA and SON |

**Core scaling pipeline** (`run-all` chains stages 1–4)

| Stage | Command | Input | Output |
|---|---|---|---|
| 1 | `bin` | paired daily CSVs | one column per temperature bin |
| 2 | `summarize` | binned CSVs | one row per bin: sum, mean, median, count, P75–P99 |
| 3 | `logtransform` | precipitation summaries | adds `<column>_log` columns |
| 4 | `scale` | both summary sets | per-percentile scaling table with regime labels |

**Modulation analysis**

| Command | Purpose |
|---|---|
| `bin-predictors` | bin precipitation, temperature and the five predictors into the same temperature bins |
| `match-predictors` | read each predictor on the day that defines each precipitation percentile |
| `regress-predictors` | regress matched predictor values on mean bin temperature |
| `cluster` | group predictor slopes by the scaling regime of the same grid cell |
| `modulation` | quantile-regression interaction models, giving the interaction coefficient and the change in pseudo-R-squared (dR2) for each predictor |

The five predictors are RH500, TCWV, VIMFC, CAPE and OMEGA500.

**Validation**

| Command | Purpose |
|---|---|
| `quantreg` | bivariate quantile regression of ln(P) on temperature, as an independent estimate of the scaling rate |

**Figures** (`tpscaling.plots`, import and call directly)

| Module | Produces |
|---|---|
| `plots.maps` | scaling-regime and peak-temperature maps |
| `plots.modulation` | interaction, dR2 and coverage panels by scaling regime |
| `plots.cluster` | mean predictor sensitivity by scaling regime, per percentile or per season |
| `plots.comparison` | percentile binning versus quantile regression, percentiles as rows and seasons as columns |

Manuscript-specific figure scripts live in `paper_figures/`.

### Regime labels

Result files written at different times spell the regimes differently
(`C-C scaling` versus `C-C like scaling`, and varying capitalisation). Every
merge and grouping passes labels through `scaling.normalize_regime`, which maps
any spelling onto four canonical labels, so files written under either
convention join correctly. New output always uses the canonical form.

### Standardised slopes in the modulation stage

`modulation` standardises temperature and every predictor within each grid cell
before fitting, so that interaction coefficients are comparable across
predictors with different units. One consequence matters: the baseline
temperature slope is then **per standard deviation of temperature, not per
degree C**. Daily temperature standard deviations over the plateau are of order
5-10 degC, so treating the standardised slope as a scaling rate overstates it by
roughly that factor and puts almost every grid cell in the super-C-C category.

The bivariate output therefore carries `temp_sd`, and the stage writes
`temp_slope_percent_per_degC` and a `scaling` regime label computed as
`temp_slope / temp_sd * 100`. Use those columns, never `temp_slope * 100`.

`dR2` is the gain in Koenker-Machado pseudo-R-squared from adding **both** the
predictor and its interaction with temperature, so it is attributable to the two
jointly rather than to the interaction alone.

### How predictors are matched to events

Within a temperature bin, the P75–P99 values each correspond to a particular
wet day. `match-predictors` finds the day whose precipitation is closest to each
percentile value and reads every predictor on that same day. The result is the
atmospheric state accompanying the events that define each percentile, not a
bin-mean of the predictor. This is why `bin-predictors` must bin all variables
together: the day-by-day correspondence is what makes the matching possible.

## Installation

```bash
git clone https://github.com/FemiPeter/tpscaling.git
cd tpscaling
pip install -r requirements.txt
```

Two stages need extra packages:

```bash
pip install "tpscaling[era5]"   # xarray, netcdf4   - for the vimfc stage
pip install "tpscaling[maps]"   # cartopy, geopandas - for the map figures
```

Python 3.9 or later. No compiled extensions, no GPU, no HPC environment
required; a full plateau-wide run is CPU-bound and single-threaded.

## Getting the input data

ERA5 predictors are retrieved and extracted with the first two stages:

```bash
python -m tpscaling era5-download --variable TCWV \
    --output-dir era5/TCWV_nc --start-year 1970 --end-year 2019

python -m tpscaling era5-extract --netcdf era5/TCWV_nc/*.nc \
    --coordinates-file data/TP_coordinates.csv --output-dir data/TCWV
```

`era5-download` needs a Copernicus Climate Data Store account and a
`~/.cdsapirc` file. Sub-daily steps are averaged to daily values by default.
VIMFC is not an ERA5 variable: retrieve `viwve` and `viwvn`, derive it with
`tpscaling vimfc`, then extract that.

The gridded station-interpolated precipitation and temperature are not produced
by this code and are not redistributed; see the Data Availability statement of
the paper.

## Input data format

Two directories with one CSV per grid cell and **matching file names** in both,
for example `column_1.csv` in each:

```
tmean/column_1.csv    prec/column_1.csv
tmean/column_2.csv    prec/column_2.csv
```

Each file is a headerless numeric matrix of daily values. The shape is not
important — values are flattened before binning — but the two files for a given
cell must have the same shape, so that element *i* of one is the same day as
element *i* of the other. In the published analysis each file holds 50 years ×
365 days for one 0.25° grid cell.

Precipitation is in mm; temperature is daily mean in degrees Celsius. Days with
precipitation below 0.1 mm are excluded as dry.

## Usage

Run every stage in order:

```bash
python -m tpscaling run-all \
    --temperature-dir   /path/to/tmean \
    --precipitation-dir /path/to/prec \
    --out-dir           /path/to/output
```

Or run a single stage:

```bash
python -m tpscaling bin   --temperature-dir ... --precipitation-dir ... --out-dir ...
python -m tpscaling scale --out-dir ... --percentiles 75 90 95 99
```

Seasonal analyses use the same commands on seasonally subset input directories:

```bash
python -m tpscaling split-seasons \
    --input-dir /path/to/prec --date-file /path/to/daily_dates.csv
```

The modulation analysis, after the core pipeline has run:

```bash
python -m tpscaling bin-predictors \
    --precipitation-dir /data/prec --temperature-dir /data/tmean \
    --predictor-dir RH500=/data/RH500    --predictor-dir TCWV=/data/TCWV \
    --predictor-dir VIMFC=/data/VIMFC    --predictor-dir CAPE=/data/CAPE \
    --predictor-dir OMEGA500=/data/OMEGA500 \
    --out-dir /path/to/output

python -m tpscaling summarize          --out-dir /path/to/output
python -m tpscaling logtransform       --out-dir /path/to/output
python -m tpscaling match-predictors   --out-dir /path/to/output
python -m tpscaling regress-predictors --out-dir /path/to/output
python -m tpscaling cluster            --out-dir /path/to/output
```

VIMFC is derived from ERA5 before binning:

```bash
python -m tpscaling vimfc \
    --eastward-file  east_flux.nc  --northward-file north_flux.nc \
    --output-file    vimfc.nc
```

## Output layout

```
output/
  tmean_bin/                      stage 1
    bin_summary/                  stage 2
  prec_bin/                       stage 1
    bin_summary/                  stages 2 and 3
  scaling/
    P75.csv  P90.csv  P95.csv  P99.csv      stage 4
  <PREDICTOR>_bin/                          bin-predictors
  matched/                                  match-predictors
  regression/                               regress-predictors
  cluster/                                  cluster
```

Each scaling table has one row per grid cell:

| column | meaning |
|---|---|
| `prefix` | grid cell identifier, matching the input file name |
| `relationship` | `monotonic` or `peak_structure` |
| `peak_side` | `left` or `right` for peak-structure cells |
| `peak_temp` | peak temperature T_peak (°C) |
| `slope` | regression slope, ln(P) per °C |
| `slope2` | scaling rate, % per °C (`slope` × 100) |
| `scaling` | regime label assigned from `slope2` |
| `r2_value`, `r_value`, `p_value`, `rmse` | fit diagnostics |
| `statistical_significance` | `TRUE` if *p* < 0.05 |
| `n_bins` | number of bins used in the fit |

## Reproducing the figures

A worked example with synthetic data is included so the pipeline can be run
without the source dataset:

```bash
python examples/make_synthetic_data.py
python -m tpscaling run-all \
    --temperature-dir   examples/synthetic/tmean \
    --precipitation-dir examples/synthetic/prec \
    --out-dir           examples/synthetic/output
```

The synthetic values are invented and carry no scientific meaning. They exist
to demonstrate that the code runs and to show the output format.

## Data availability

The station-interpolated gridded daily precipitation and temperature dataset
and the ERA5 reanalysis predictors are not redistributed here. See the Data
Availability statement of the associated paper for their sources.

## Parameters

Defaults match the published analysis and are defined at the top of each
module:

| parameter | default | module |
|---|---|---|
| wet-day threshold | 0.1 mm | `binning.WET_DAY_THRESHOLD` |
| bin edges | 0–100 % in steps of 5 (20 bins) | `binning.BIN_EDGES_PERCENT` |
| LOESS fraction | 0.3 | `scaling.LOESS_FRAC` |
| minimum bins for a fit | 10 | `scaling.MIN_BINS` |
| significance level | 0.05 | `scaling.ALPHA` |
| regime bounds | 0, 5, 9 % °C⁻¹ | `scaling.REGIME_BOUNDS` |
| quantiles | 0.75, 0.90, 0.95, 0.99 | `quantreg.QUANTILES` |
| cross-validation folds | 5, contiguous | `quantreg.N_FOLDS` |
| VIMFC discretisation | `"simple"` | `vimfc.compute_vimfc` |
| minimum predictor coverage drawn | 2 % | `plots.modulation.MIN_COVERAGE_PERCENT` |

### Notes on two choices

**Cross-validation folds are contiguous blocks of the record, not random
draws.** `KFold` is used without shuffling, so the spread across folds reflects
variation between periods rather than the sampling noise of random subsets.

**VIMFC has two discretisations.** `method="simple"` reproduces the published
analysis: the zonal term carries the 1/cos(φ) metric factor, the meridional
term does not include cos(φ) inside the derivative. `method="spherical"`
applies the exact spherical divergence. Over the plateau the two differ by
about a percent. Latitude is sorted ascending before differentiating in both
cases, since ERA5 ships it descending.

## Citation

If you use this code, please cite both the paper and the archived release. See
`CITATION.cff`.

## License

MIT. See `LICENSE`.
