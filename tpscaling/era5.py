"""
ERA5 retrieval and extraction to per-grid-cell CSV.

Two steps sit upstream of everything else in this package.

**download** - retrieve one ERA5 single-level or pressure-level variable over
the Tibetan Plateau, one NetCDF file per year, through the Copernicus Climate
Data Store API. Requires a CDS account and a ``~/.cdsapirc`` file; see
https://cds.climate.copernicus.eu/how-to-api.

**extract** - sample a NetCDF field at each plateau grid cell and write one
headerless CSV per cell, named ``column_<n>.csv``. This is the format every
later stage expects: one value per row, in time order, with no header.

Grid cells are listed in ``data/TP_coordinates.csv``: 4,019 points on a 0.25
degree grid, longitude in the first column and latitude in the second, in the
order used throughout the analysis. Cell *n* in that file is ``column_n``,
which is what ties the extracted series, the scaling tables and the maps
together.

Nearest-neighbour selection is used, so the ERA5 cell whose centre is closest
to each listed point is taken. On a 0.25 degree ERA5 grid matching a 0.25
degree target grid this is an exact match to within floating-point tolerance,
but no interpolation is performed and none is implied.

Requires ``xarray``, ``netcdf4`` and, for downloading, ``cdsapi``:

    pip install "tpscaling[era5]"
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

#: Bounding box used for every ERA5 retrieval: north, west, south, east.
TP_AREA = (42, 73, 24, 106)

#: Sub-daily times retrieved, averaged to daily values downstream.
TIMES = ("00:00", "06:00", "12:00", "18:00")

#: Single-level variables used in the analysis, by short name.
SINGLE_LEVEL_VARIABLES = {
    "TCWV": "total_column_water_vapour",
    "CAPE": "convective_available_potential_energy",
    "viwve": "vertical_integral_of_eastward_water_vapour_flux",
    "viwvn": "vertical_integral_of_northward_water_vapour_flux",
}

#: Pressure-level variables, retrieved at 500 hPa.
PRESSURE_LEVEL_VARIABLES = {
    "RH500": "relative_humidity",
    "OMEGA500": "vertical_velocity",
}


def download(variable: str, output_dir, years, area=TP_AREA, times=TIMES,
             pressure_level: str | None = None, verbose: bool = True):
    """Retrieve one ERA5 variable, one NetCDF file per year.

    ``variable`` is a CDS variable name, or a short name from
    :data:`SINGLE_LEVEL_VARIABLES` or :data:`PRESSURE_LEVEL_VARIABLES`.
    """
    try:
        import cdsapi
    except ImportError as exc:                                # pragma: no cover
        raise ImportError('downloading requires cdsapi: '
                          'pip install "tpscaling[era5]"') from exc

    if variable in SINGLE_LEVEL_VARIABLES:
        dataset = "reanalysis-era5-single-levels"
        cds_name = SINGLE_LEVEL_VARIABLES[variable]
    elif variable in PRESSURE_LEVEL_VARIABLES:
        dataset = "reanalysis-era5-pressure-levels"
        cds_name = PRESSURE_LEVEL_VARIABLES[variable]
        pressure_level = pressure_level or "500"
    else:
        dataset = ("reanalysis-era5-pressure-levels" if pressure_level
                   else "reanalysis-era5-single-levels")
        cds_name = variable

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    for year in years:
        target = output_dir / f"{year}.nc"
        if target.exists():
            if verbose:
                print(f"  {target.name} exists, skipping")
            continue
        request = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": cds_name,
            "year": str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": list(times),
            "area": list(area),
        }
        if pressure_level:
            request["pressure_level"] = str(pressure_level)
        client.retrieve(dataset, request, str(target))
        if verbose:
            print(f"  downloaded {year} -> {target}")
    return output_dir


def load_coordinates(coordinates_file, id_prefix: str = "column"):
    """Read the plateau coordinate list.

    Accepts the two-column headerless form (longitude, latitude) used in
    ``data/TP_coordinates.csv``, or a file with ``name``, ``long`` and ``lat``
    columns. Either way the result has ``name``, ``long`` and ``lat``, with
    names generated as ``<id_prefix>_<n>`` from the row order when absent.
    """
    probe = pd.read_csv(coordinates_file, nrows=1, header=None)
    first = str(probe.iloc[0, 0]).strip().lower()
    has_header = first in {"name", "long", "longitude", "lon", "x"}

    if has_header:
        table = pd.read_csv(coordinates_file)
        columns = {c.strip().lower(): c for c in table.columns}
        lon = columns.get("long") or columns.get("longitude") or columns.get("lon")
        lat = columns.get("lat") or columns.get("latitude")
        if lon is None or lat is None:
            raise ValueError(
                f"{coordinates_file} has headers but no recognisable "
                f"longitude/latitude columns: {list(table.columns)}")
        name = columns.get("name")
        names = (table[name].astype(str) if name else
                 pd.Series([f"{id_prefix}_{i + 1}" for i in range(len(table))]))
        return pd.DataFrame({"name": names.to_numpy(),
                             "long": pd.to_numeric(table[lon]).to_numpy(),
                             "lat": pd.to_numeric(table[lat]).to_numpy()})

    table = pd.read_csv(coordinates_file, header=None)
    if table.shape[1] < 2:
        raise ValueError(
            f"{coordinates_file} must have at least two columns "
            f"(longitude, latitude); found {table.shape[1]}")
    return pd.DataFrame({
        "name": [f"{id_prefix}_{i + 1}" for i in range(len(table))],
        "long": pd.to_numeric(table.iloc[:, 0]).to_numpy(),
        "lat": pd.to_numeric(table.iloc[:, 1]).to_numpy(),
    })


def extract(netcdf_files, coordinates_file, output_dir, variable=None,
            daily: str | None = "mean", time_dim: str | None = None,
            verbose: bool = True) -> int:
    """Sample a NetCDF field at every plateau grid cell.

    ``netcdf_files`` is one path or a list of paths, opened together along the
    time dimension. ``variable`` names the field; if omitted, the only data
    variable is used. ``daily`` resamples sub-daily steps to daily values
    (``"mean"``, ``"sum"``, ``"max"``, or ``None`` to keep the native
    frequency).

    Writes ``<name>.csv`` per grid cell: one value per row, no header.
    """
    try:
        import xarray as xr
    except ImportError as exc:                                # pragma: no cover
        raise ImportError('extraction requires xarray and netcdf4: '
                          'pip install "tpscaling[era5]"') from exc

    if isinstance(netcdf_files, (str, os.PathLike)):
        dataset = xr.open_dataset(netcdf_files)
    else:
        dataset = xr.open_mfdataset(sorted(str(p) for p in netcdf_files),
                                    combine="by_coords")

    if variable is None:
        candidates = [v for v in dataset.data_vars
                      if dataset[v].ndim >= 3 or "time" in dataset[v].dims]
        if len(candidates) != 1:
            raise ValueError(
                f"specify variable; the file holds {list(dataset.data_vars)}")
        variable = candidates[0]
    field = dataset[variable]

    if time_dim is None:
        for candidate in ("valid_time", "time"):
            if candidate in field.dims:
                time_dim = candidate
                break
    if time_dim is None:
        raise ValueError(f"no time dimension found in {field.dims}")

    if daily:
        field = getattr(field.resample({time_dim: "1D"}), daily)()

    points = load_coordinates(coordinates_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # One vectorised selection over all points, rather than one call per point:
    # the per-point loop in the original script reopened the index 4,019 times.
    selector = {
        "longitude": xr.DataArray(points["long"].to_numpy(), dims="cell"),
        "latitude": xr.DataArray(points["lat"].to_numpy(), dims="cell"),
    }
    sampled = field.sel(**selector, method="nearest").load()

    values = sampled.transpose(time_dim, "cell").to_numpy()
    for index, name in enumerate(points["name"]):
        pd.DataFrame(values[:, index]).to_csv(
            output_dir / f"{name}.csv", header=False, index=False)

    if verbose:
        print(f"extracted {variable} at {len(points)} grid cells "
              f"({values.shape[0]} time steps) -> {output_dir}")
    return len(points)
