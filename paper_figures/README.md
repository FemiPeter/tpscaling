# Paper figure scripts

Standalone scripts for specific manuscript figures. Unlike the `tpscaling`
package, these are tied to the layout of the published figures and read paths
configured at the top of each file.

| Script | Produces |
|---|---|
| `table_data.py` | writes `scaling_tables.xlsx` from the published coverage, mean-rate and peak-structure values |
| `table_figures.py` | annotated-matrix figures replacing Tables 1 and 2 |
| `composite_curves.py` | composite ln(P)-temperature curves by scaling regime and structure type |

`composite_curves.py` has its own `CONFIG` block; edit `ROOT` and `SEASON_DIRS`
before running. The other two read `scaling_tables.xlsx` in this directory.
