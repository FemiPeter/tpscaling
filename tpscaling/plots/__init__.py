"""
Figure scripts.

Submodules
----------
``maps``        scaling-regime and peak-temperature maps over the plateau
``modulation``  modulation, dR2 and proportion panels by scaling regime
``cluster``     mean predictor slope by scaling regime
``comparison``  percentile binning versus quantile regression

Mapping needs ``cartopy`` and ``geopandas``; everything else needs only
``matplotlib`` and ``seaborn``:

    pip install "tpscaling[maps]"
"""

#: Scaling-regime colours, shared by every figure.
REGIME_COLOURS = {
    "Negative Scaling": "#ff7855",
    "Sub C-C scaling": "#75bbfd",
    "C-C scaling": "#3e82fc",
    "Super C-C scaling": "#0652ff",
}

#: Regime order, coldest-scaling first.
REGIME_ORDER = tuple(REGIME_COLOURS)


def normalize_regime(label):
    """Canonical regime label; re-exported so figure code need not import
    from :mod:`tpscaling.scaling`."""
    from ..scaling import normalize_regime as _normalize
    return _normalize(label)

#: Predictor display order, as used in the paper figures.
PREDICTOR_ORDER = ("TCWV", "RH500", "CAPE", "OMEGA500", "VIMFC")

#: Units of each predictor's temperature sensitivity, for axis labels.
PREDICTOR_UNITS = {
    "TCWV": "kg m$^{-2}$ $^{\\circ}$C$^{-1}$",
    "RH500": "% $^{\\circ}$C$^{-1}$",
    "CAPE": "J kg$^{-1}$ $^{\\circ}$C$^{-1}$",
    "OMEGA500": "Pa s$^{-1}$ $^{\\circ}$C$^{-1}$",
    "VIMFC": "kg m$^{-2}$ s$^{-1}$ $^{\\circ}$C$^{-1}$",
}

SEASONS = ("Annual", "Winter", "Spring", "Summer", "Autumn")
PERCENTILES = (75, 90, 95, 99)

__all__ = ["REGIME_COLOURS", "REGIME_ORDER", "PREDICTOR_ORDER",
           "PREDICTOR_UNITS", "SEASONS", "PERCENTILES", "normalize_regime"]
