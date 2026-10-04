# Data

## TP_coordinates.csv

The 4,019 Tibetan Plateau grid cells used throughout the analysis, on a 0.25
degree grid. Two columns, no header: longitude, then latitude. Row *n* is grid
cell `column_n`, and that correspondence ties together the extracted ERA5
series, the scaling tables and the maps.

Longitude spans 73.625 to 104.370 degE, latitude 26.125 to 39.625 degN.

## tp_roi.rar

Tibetan Plateau region-of-interest shapefile, used to clip grid cells and draw
the boundary in the map figures. 

## Not included

The gridded station-interpolated daily precipitation and temperature dataset is
not redistributed here. See the Data Availability statement of the associated
paper.

ERA5 fields are not redistributed either; `tpscaling era5-download` retrieves
them from the Copernicus Climate Data Store.
