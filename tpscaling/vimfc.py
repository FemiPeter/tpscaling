"""
Vertically integrated moisture flux convergence (VIMFC) from ERA5.

VIMFC is the negative divergence of the vertically integrated horizontal
moisture flux, computed from the ERA5 fields ``viwve`` (eastward) and
``viwvn`` (northward):

    VIMFC = -( 1/(a cos(phi)) * d(viwve)/d(lambda)
             + 1/a           * d(viwvn)/d(phi) )          [method="simple"]

Positive values mean moisture converging into the column.

Two discretisations are available. ``method="simple"`` is the one used in the
published analysis: the zonal term carries the 1/cos(phi) metric factor but the
meridional term does not include the cos(phi) weighting inside the derivative.
``method="spherical"`` applies the full spherical-divergence form,

    1/(a cos(phi)) * [ d(viwve)/d(lambda) + d(viwvn * cos(phi))/d(phi) ]

which is formally exact. Over the Tibetan Plateau the two differ by roughly a
percent, far less than the uncertainty in the fluxes themselves, but the
spherical form is preferred for new work.

Latitude is sorted ascending before differentiating. ERA5 ships latitude in
descending order, and differentiating a descending coordinate without sorting
flips the sign of the meridional term.

Requires ``xarray`` and ``netcdf4``, installed via the ``era5`` extra:

    pip install "tpscaling[era5]"
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS = 6_371_000.0   #: Earth radius (m)
DEG2RAD = np.pi / 180.0


def compute_vimfc(eastward_file, northward_file, output_file=None,
                  eastward_var: str = "viwve", northward_var: str = "viwvn",
                  method: str = "simple"):
    """Compute VIMFC from ERA5 vertically integrated moisture flux components.

    Parameters
    ----------
    eastward_file, northward_file
        NetCDF files holding ``viwve`` and ``viwvn`` (kg m-1 s-1).
    output_file
        If given, the result is written here as NetCDF.
    method
        ``"simple"`` reproduces the published analysis; ``"spherical"`` applies
        the full spherical divergence.

    Returns
    -------
    xarray.DataArray
        VIMFC in kg m-2 s-1, positive for convergence.
    """
    try:
        import xarray as xr                                   # noqa: F401
    except ImportError as exc:                                # pragma: no cover
        raise ImportError(
            "computing VIMFC requires xarray and netcdf4: "
            'pip install "tpscaling[era5]"') from exc
    import xarray as xr

    if method not in ("simple", "spherical"):
        raise ValueError(f"unknown method: {method!r}")

    eastward = xr.open_dataset(eastward_file)[eastward_var]
    northward = xr.open_dataset(northward_file)[northward_var]

    if eastward.dims != northward.dims:
        raise ValueError(
            f"dimension mismatch: {eastward.dims} vs {northward.dims}")

    # ERA5 latitude runs north to south; sort ascending before differentiating.
    eastward = eastward.sortby("latitude")
    northward = northward.sortby("latitude")

    cos_lat = np.cos(eastward["latitude"] * DEG2RAD)

    d_east_dlon = eastward.differentiate("longitude")       # per degree
    div_east = d_east_dlon / (EARTH_RADIUS * cos_lat * DEG2RAD)

    if method == "simple":
        d_north_dlat = northward.differentiate("latitude")
        div_north = d_north_dlat / (EARTH_RADIUS * DEG2RAD)
    else:
        d_north_dlat = (northward * cos_lat).differentiate("latitude")
        div_north = d_north_dlat / (EARTH_RADIUS * cos_lat * DEG2RAD)

    vimfc = -(div_east + div_north)
    vimfc.name = "vimfc"
    vimfc.attrs = {
        "long_name": "Vertically integrated moisture flux convergence",
        "units": "kg m-2 s-1",
        "description": "Negative divergence of the vertically integrated "
                       "moisture flux; positive means convergence",
        "discretisation": method,
        "source_files": f"{eastward_file}, {northward_file}",
    }

    if output_file is not None:
        vimfc.to_netcdf(output_file)
        print(f"VIMFC written to {output_file}")
    return vimfc
