"""
Maps of scaling regime and peak temperature over the Tibetan Plateau.

Grid cells are located from a coordinate file. If that file carries a cell
identifier matching the ``prefix`` column of the scaling tables, the two are
merged on it; otherwise they are merged on row position, which is the
assumption the original workflow relied on. Merging on an identifier is
preferred: a positional merge fails silently if either file is ever reordered.

Pass ``id_column`` to name the identifier column, or leave it ``None`` to let
the loader look for a column whose values look like ``column_<n>``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import PERCENTILES, REGIME_COLOURS, REGIME_ORDER, normalize_regime

TPEAK_RANGE = (-30, 30)


def _imports():
    try:
        import cartopy.crs as ccrs
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
    except ImportError as exc:                                # pragma: no cover
        raise ImportError('mapping requires cartopy and geopandas: '
                          'pip install "tpscaling[maps]"') from exc
    return ccrs, gpd, plt, LongitudeFormatter, LatitudeFormatter


def load_coordinates(coordinates_file, id_column=None, lon_column=0,
                     lat_column=1):
    """Read the coordinate file as a GeoDataFrame with a ``cell_id`` column.

    ``cell_id`` is taken from ``id_column`` if given, from the first column
    holding ``column_<n>`` strings if one exists, and otherwise built from the
    row position as ``column_<i+1>``, reproducing the positional convention.
    """
    _, gpd, _, _, _ = _imports()

    header = 0 if id_column is not None else None
    table = pd.read_csv(coordinates_file, header=header)

    if id_column is not None:
        identifiers = table[id_column].astype(str)
    else:
        identifiers = None
        for column in table.columns:
            values = table[column].astype(str)
            if values.str.match(r"^column_\d+$").all():
                identifiers = values
                break
        if identifiers is None:
            identifiers = pd.Series(
                [f"column_{i + 1}" for i in range(len(table))],
                index=table.index)

    longitude = pd.to_numeric(table.iloc[:, lon_column], errors="coerce")
    latitude = pd.to_numeric(table.iloc[:, lat_column], errors="coerce")
    frame = gpd.GeoDataFrame(
        {"cell_id": identifiers.to_numpy()},
        geometry=gpd.points_from_xy(longitude, latitude), crs="EPSG:4326")
    return frame


def _merge(points, table, id_column_in_table="prefix"):
    """Merge scaling results onto coordinates by cell id, with a size check."""
    if id_column_in_table not in table.columns:
        raise ValueError(
            f"scaling table has no {id_column_in_table!r} column to merge on")
    merged = points.merge(table, left_on="cell_id",
                          right_on=id_column_in_table, how="inner")
    if merged.empty:
        raise ValueError(
            "no grid cells matched between the coordinate file and the "
            "scaling table; check that cell identifiers use the same form")
    return merged


def _decorate(axis, boundary, ccrs, LongitudeFormatter, LatitudeFormatter,
              centre_ticks=False, labelsize=14):
    boundary.plot(ax=axis, facecolor="none", edgecolor="black", linewidth=0.5,
                  transform=ccrs.PlateCarree())
    west, south, east, north = boundary.total_bounds
    x = np.linspace(west, east, 7)
    y = np.linspace(south, north, 8)
    if centre_ticks:
        x = (x[1:] + x[:-1]) / 2
        y = (y[1:] + y[:-1]) / 2
    axis.set_xticks(np.round(x).astype(int), crs=ccrs.PlateCarree())
    axis.set_yticks(np.round(y).astype(int), crs=ccrs.PlateCarree())
    axis.xaxis.set_major_formatter(LongitudeFormatter())
    axis.yaxis.set_major_formatter(LatitudeFormatter())
    axis.tick_params(axis="x", top=True, bottom=False, labeltop=True,
                     labelbottom=False, labelsize=labelsize)
    axis.tick_params(axis="y", left=True, right=False, labelleft=True,
                     labelright=False, labelsize=labelsize)


# ------------------------------------------------------- SCALING REGIME MAP
def plot_scaling_regimes(scaling_dir, coordinates_file, boundary_file,
                         output_file, season_label: str = "",
                         percentiles=PERCENTILES, id_column=None,
                         dpi: int = 500, marker_size: float = 30.0):
    """Four-panel map of scaling regime, with an inset bar chart of coverage.

    Grid cells with a peak structure are overlaid with a plus sign.
    """
    ccrs, gpd, plt, LongitudeFormatter, LatitudeFormatter = _imports()
    from matplotlib.colors import BoundaryNorm, ListedColormap

    codes = {label: i for i, label in enumerate(REGIME_ORDER)}
    cmap = ListedColormap([REGIME_COLOURS[label] for label in REGIME_ORDER])
    norm = BoundaryNorm(list(range(len(REGIME_ORDER) + 1)), cmap.N)

    boundary = gpd.read_file(boundary_file)
    points = load_coordinates(coordinates_file, id_column)
    boundary = boundary.to_crs(points.crs)
    clipped = gpd.clip(points, boundary)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9.5),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    axes = axes.flatten()
    scatter = None

    for index, percentile in enumerate(percentiles):
        table = pd.read_csv(Path(scaling_dir) / f"P{percentile}.csv")
        merged = _merge(clipped, table)
        axis = axes[index]

        scatter = axis.scatter(
            merged.geometry.x, merged.geometry.y,
            c=merged["scaling"].map(normalize_regime).map(codes),
            cmap=cmap, norm=norm,
            s=marker_size, alpha=0.8, transform=ccrs.PlateCarree())
        peaks = merged["relationship"] == "peak_structure"
        axis.scatter(merged.loc[peaks].geometry.x,
                     merged.loc[peaks].geometry.y, marker="+", c="black",
                     s=10, linewidths=0.4, alpha=0.6,
                     transform=ccrs.PlateCarree())

        axis.set_title(f"({'abcd'[index]}) P{percentile} {season_label}".strip(),
                       fontsize=22, fontweight="bold", loc="left")
        _decorate(axis, boundary, ccrs, LongitudeFormatter, LatitudeFormatter,
                  centre_ticks=True, labelsize=16)

        shares = (merged["scaling"].map(normalize_regime)
                  .value_counts(normalize=True)
                  .reindex(REGIME_ORDER, fill_value=0) * 100).round(1)
        inset = fig.add_axes([0.128 + (index % 2) * 0.422,
                              0.552 - (index // 2) * 0.42, 0.08, 0.1])
        bars = inset.bar(range(len(REGIME_ORDER)), shares.to_numpy(),
                         color=cmap.colors, edgecolor="black", width=0.6)
        inset.set_ylim(0, 100)
        inset.set_xticks([]); inset.set_yticks([])
        inset.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, shares):
            inset.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 7,
                       f"{value}%", ha="center", fontsize=11, rotation=90)

    for index in range(len(percentiles), 4):
        axes[index].set_visible(False)

    bar_axis = fig.add_axes([0.125, 0.04, 0.775, 0.02])
    colourbar = plt.colorbar(scatter, cax=bar_axis, orientation="horizontal")
    colourbar.set_ticks([i + 0.5 for i in range(len(REGIME_ORDER))])
    colourbar.set_ticklabels(list(REGIME_ORDER))
    colourbar.ax.tick_params(labelsize=18)
    colourbar.set_label("Scaling regime", fontsize=20)
    fig.text(0.5, 0.085, "+ = peak structure", fontsize=18, ha="center")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_file}")
    return output_file


# ----------------------------------------------------- PEAK TEMPERATURE MAP
def load_colormap(path=None):
    """Build a colormap from a JSON mapping, or fall back to a diverging one."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if path is None:
        return plt.get_cmap("RdYlBu_r")
    with open(path) as handle:
        return ListedColormap(list(json.load(handle).values()))


def plot_peak_temperature(scaling_dir, coordinates_file, boundary_file,
                          output_file, season_label: str = "",
                          percentiles=PERCENTILES, colormap_file=None,
                          id_column=None, value_range=TPEAK_RANGE,
                          dpi: int = 500, marker_size: float = 4.0):
    """Four-panel map of peak temperature for peak-structure grid cells."""
    ccrs, gpd, plt, LongitudeFormatter, LatitudeFormatter = _imports()
    from matplotlib.colors import Normalize

    boundary = gpd.read_file(boundary_file)
    points = load_coordinates(coordinates_file, id_column)
    boundary = boundary.to_crs(points.crs)
    clipped = gpd.clip(points, boundary)

    cmap = load_colormap(colormap_file)
    norm = Normalize(vmin=value_range[0], vmax=value_range[1])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                             subplot_kw={"projection": ccrs.PlateCarree()},
                             gridspec_kw={"hspace": 0.15, "wspace": 0.25,
                                          "top": 0.95})
    scatter = None
    for index, percentile in enumerate(percentiles):
        table = pd.read_csv(Path(scaling_dir) / f"P{percentile}.csv")
        merged = _merge(clipped, table)
        share = 100.0 * (merged["relationship"] == "peak_structure").mean()
        peaks = merged[merged["relationship"] == "peak_structure"]

        axis = axes.flat[index]
        scatter = axis.scatter(peaks.geometry.x, peaks.geometry.y,
                               c=peaks["peak_temp"], cmap=cmap, norm=norm,
                               s=marker_size, alpha=0.8,
                               transform=ccrs.PlateCarree())
        axis.set_title(f"({'abcd'[index]}) P{percentile} {season_label}".strip(),
                       fontsize=18, weight="bold", loc="left")
        _decorate(axis, boundary, ccrs, LongitudeFormatter, LatitudeFormatter,
                  labelsize=13)
        axis.text(0.035, 0.07, f"P.S = {share:.1f}%", transform=axis.transAxes,
                  fontsize=13,
                  bbox=dict(facecolor="white", edgecolor="black",
                            boxstyle="round,pad=0.3"))

    colourbar = fig.colorbar(scatter, ax=axes.ravel().tolist(),
                             orientation="horizontal", pad=0.05, aspect=50)
    colourbar.set_label("Peak temperature (\u00b0C)", fontsize=18)
    colourbar.ax.tick_params(labelsize=16)
    colourbar.set_ticks(np.arange(value_range[0], value_range[1] + 1, 5))

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_file}")
    return output_file
