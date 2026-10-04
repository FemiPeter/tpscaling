"""Command-line interface: ``python -m tpscaling <stage> ...``"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import PERCENTILES, __version__
from . import (binning, era5, modulation, predictors, quantreg, scaling,
               seasons, summarize)


def _layout(root: Path) -> dict:
    """Standard output layout under one working directory."""
    return {
        "temp_bin": root / "tmean_bin",
        "prec_bin": root / "prec_bin",
        "temp_summary": root / "tmean_bin" / "bin_summary",
        "prec_summary": root / "prec_bin" / "bin_summary",
        "scaling": root / "scaling",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="tpscaling",
        description="Extreme precipitation-temperature scaling pipeline.")
    parser.add_argument("--version", action="version",
                        version=f"tpscaling {__version__}")
    sub = parser.add_subparsers(dest="stage", required=True)

    p1 = sub.add_parser("bin", help="stage 1: temperature-percentile binning")
    p1.add_argument("--temperature-dir", required=True)
    p1.add_argument("--precipitation-dir", required=True)
    p1.add_argument("--out-dir", required=True)
    p1.add_argument("--wet-day-threshold", type=float,
                    default=binning.WET_DAY_THRESHOLD)

    p2 = sub.add_parser("summarize", help="stage 2: per-bin statistics")
    p2.add_argument("--out-dir", required=True)

    p3 = sub.add_parser("logtransform", help="stage 3: ln(P) columns")
    p3.add_argument("--out-dir", required=True)

    p4 = sub.add_parser("scale", help="stage 4: scaling fit and classification")
    p4.add_argument("--out-dir", required=True)
    p4.add_argument("--percentiles", type=int, nargs="+",
                    default=list(PERCENTILES))
    p4.add_argument("--loess-frac", type=float, default=scaling.LOESS_FRAC)

    ps = sub.add_parser("split-seasons", help="split daily series by season")
    ps.add_argument("--input-dir", required=True)
    ps.add_argument("--date-file", required=True)
    ps.add_argument("--output-dir", default=None)

    pd_ = sub.add_parser("era5-download", help="retrieve ERA5 over the plateau")
    pd_.add_argument("--variable", required=True)
    pd_.add_argument("--output-dir", required=True)
    pd_.add_argument("--start-year", type=int, required=True)
    pd_.add_argument("--end-year", type=int, required=True)
    pd_.add_argument("--pressure-level", default=None)

    pe = sub.add_parser("era5-extract",
                        help="sample a NetCDF field at the plateau grid cells")
    pe.add_argument("--netcdf", nargs="+", required=True)
    pe.add_argument("--coordinates-file", required=True)
    pe.add_argument("--output-dir", required=True)
    pe.add_argument("--variable", default=None)
    pe.add_argument("--daily", default="mean",
                    choices=["mean", "sum", "max", "none"])

    pv = sub.add_parser("vimfc", help="compute VIMFC from ERA5 moisture fluxes")
    pv.add_argument("--eastward-file", required=True)
    pv.add_argument("--northward-file", required=True)
    pv.add_argument("--output-file", required=True)
    pv.add_argument("--method", choices=["simple", "spherical"],
                    default="simple")

    pq = sub.add_parser("quantreg", help="bivariate quantile regression")
    pq.add_argument("--temperature-dir", required=True)
    pq.add_argument("--precipitation-dir", required=True)
    pq.add_argument("--output-dir", required=True)
    pq.add_argument("--n-folds", type=int, default=quantreg.N_FOLDS)

    pm = sub.add_parser("bin-predictors",
                        help="bin precipitation, temperature and predictors")
    pm.add_argument("--precipitation-dir", required=True)
    pm.add_argument("--temperature-dir", required=True)
    pm.add_argument("--predictor-dir", action="append", default=[],
                    metavar="NAME=PATH",
                    help="repeatable, e.g. --predictor-dir RH500=/data/RH500")
    pm.add_argument("--out-dir", required=True)

    px = sub.add_parser("match-predictors",
                        help="read predictors on percentile-defining days")
    px.add_argument("--out-dir", required=True)
    px.add_argument("--predictors", nargs="+", default=list(predictors.PREDICTORS))

    pr = sub.add_parser("regress-predictors",
                        help="regress matched predictors on temperature")
    pr.add_argument("--out-dir", required=True)
    pr.add_argument("--predictors", nargs="+", default=list(predictors.PREDICTORS))

    pc = sub.add_parser("cluster",
                        help="group predictor slopes by scaling regime")
    pc.add_argument("--out-dir", required=True)
    pc.add_argument("--predictors", nargs="+", default=list(predictors.PREDICTORS))

    pz = sub.add_parser("modulation",
                        help="quantile-regression interaction models and dR2")
    pz.add_argument("--precipitation-dir", required=True)
    pz.add_argument("--temperature-dir", required=True)
    pz.add_argument("--predictor-dir", action="append", default=[],
                    metavar="NAME=PATH",
                    help="repeatable, e.g. --predictor-dir RH500=/data/RH500")
    pz.add_argument("--output-dir", required=True)

    pa = sub.add_parser("run-all", help="stages 1-4 in order")
    pa.add_argument("--temperature-dir", required=True)
    pa.add_argument("--precipitation-dir", required=True)
    pa.add_argument("--out-dir", required=True)
    pa.add_argument("--percentiles", type=int, nargs="+",
                    default=list(PERCENTILES))
    pa.add_argument("--wet-day-threshold", type=float,
                    default=binning.WET_DAY_THRESHOLD)
    pa.add_argument("--loess-frac", type=float, default=scaling.LOESS_FRAC)

    args = parser.parse_args(argv)

    if args.stage == "split-seasons":
        seasons.run(args.input_dir, args.date_file, args.output_dir)
        return 0

    if args.stage == "era5-download":
        era5.download(args.variable, args.output_dir,
                      range(args.start_year, args.end_year + 1),
                      pressure_level=args.pressure_level)
        return 0

    if args.stage == "era5-extract":
        era5.extract(args.netcdf, args.coordinates_file, args.output_dir,
                     variable=args.variable,
                     daily=None if args.daily == "none" else args.daily)
        return 0

    if args.stage == "vimfc":
        from . import vimfc
        vimfc.compute_vimfc(args.eastward_file, args.northward_file,
                            args.output_file, method=args.method)
        return 0

    if args.stage == "modulation":
        named = {}
        for item in args.predictor_dir:
            if "=" not in item:
                parser.error(f"--predictor-dir expects NAME=PATH, got {item!r}")
            name, path = item.split("=", 1)
            named[name] = path
        if not named:
            parser.error("modulation needs at least one --predictor-dir")
        modulation.run(args.precipitation_dir, args.temperature_dir, named,
                       args.output_dir)
        return 0

    if args.stage == "quantreg":
        quantreg.run(args.temperature_dir, args.precipitation_dir,
                     args.output_dir, n_folds=args.n_folds)
        return 0

    paths = _layout(Path(args.out_dir))

    if args.stage == "bin-predictors":
        named = {}
        for item in args.predictor_dir:
            if "=" not in item:
                parser.error(f"--predictor-dir expects NAME=PATH, got {item!r}")
            name, path = item.split("=", 1)
            named[name] = path
        variable_dirs = {"precipitation": args.precipitation_dir,
                         "temperature": args.temperature_dir, **named}
        out_root = Path(args.out_dir)
        output_dirs = {"precipitation": out_root / "prec_bin",
                       "temperature": out_root / "tmean_bin",
                       **{n: out_root / f"{n}_bin" for n in named}}
        binning.run_multivariable(variable_dirs, output_dirs)
        return 0

    if args.stage == "match-predictors":
        root = Path(args.out_dir)
        predictors.run_match(
            paths["prec_summary"], root / "prec_bin",
            {n: root / f"{n}_bin" for n in args.predictors},
            root / "matched")
        return 0

    if args.stage == "regress-predictors":
        root = Path(args.out_dir)
        predictors.run_regression(paths["temp_summary"], root / "matched",
                                  root / "regression",
                                  predictors=args.predictors)
        return 0

    if args.stage == "cluster":
        root = Path(args.out_dir)
        predictors.run_cluster(root / "regression", paths["scaling"],
                               root / "cluster" / "predictor_slopes_by_regime.csv",
                               predictors=args.predictors)
        return 0

    if args.stage in ("bin", "run-all"):
        print("stage 1: binning")
        binning.run(args.temperature_dir, args.precipitation_dir,
                    paths["temp_bin"], paths["prec_bin"],
                    threshold=args.wet_day_threshold)

    if args.stage in ("summarize", "run-all"):
        print("stage 2: summarizing")
        summarize.run_summary(paths["temp_bin"], paths["temp_summary"])
        summarize.run_summary(paths["prec_bin"], paths["prec_summary"])

    if args.stage in ("logtransform", "run-all"):
        print("stage 3: log transform")
        summarize.run_log_transform(paths["prec_summary"])

    if args.stage in ("scale", "run-all"):
        print("stage 4: scaling fit")
        scaling.run(paths["temp_summary"], paths["prec_summary"],
                    paths["scaling"], percentiles=args.percentiles,
                    loess_frac=getattr(args, "loess_frac", scaling.LOESS_FRAC))

    print(f"\ndone -> {Path(args.out_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
