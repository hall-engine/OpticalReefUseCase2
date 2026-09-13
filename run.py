#!/usr/bin/env python3
"""
run.py
======
Command-line driver for the Optical Reef direct-imaging yield pipeline.

Fast local test (subsamples the cached DR3 catalog, seconds to run):

    python run.py --test

Full local run against the whole cached catalog:

    python run.py --out_dir results_full

The catalog (gaia_pole_sample.csv) is produced separately by fetch_local.py.

Common knobs:
    --n_draws N          orbital draws per star            [config default 2000]
    --max_stars N        cap selected stars in --test      [config default 200]
    --mode M             detection | characterization      [detection]
    --exp_detect_min M   detection integration [minutes]   [5]
    --exp_char_min M     characterization integration [min][360]
    --spectral_R R       resolution for characterization   [70]
    --apertures ...      aperture diameters [m]            [4 6 8 ... 1000]
    --eta_earth F        HZ occurrence rate                [0.24]
    --no_plots           skip figure generation
"""

from __future__ import annotations

import argparse
import os

from config import RunConfig
import pipeline


def build_config(args) -> RunConfig:
    cfg = RunConfig()
    if args.n_draws is not None:
        cfg.montecarlo.n_draws = args.n_draws
    if args.max_stars is not None:
        cfg.survey.test_max_stars = args.max_stars
    if args.apertures is not None:
        cfg.apertures_m = args.apertures
    if args.eta_earth is not None:
        cfg.drake.eta_earth = args.eta_earth
    if args.k_owa is not None:
        cfg.instrument.k_owa = None if args.k_owa == 0 else args.k_owa
    if args.catalog is not None:
        cfg.survey.catalog_path = args.catalog
    # observation mode
    cfg.observation.mode = args.mode
    if args.spectral_R is not None:
        cfg.observation.spectral_R = args.spectral_R
    if args.snr_detect is not None:
        cfg.observation.snr_detect = args.snr_detect
    if args.snr_char is not None:
        cfg.observation.snr_char = args.snr_char
    if args.exp_detect_min is not None:
        cfg.observation.exposure_detect_s = args.exp_detect_min * 60.0
    if args.exp_char_min is not None:
        cfg.observation.exposure_char_s = args.exp_char_min * 60.0
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", action="store_true",
                   help="fast mode: subsample the catalog to --max_stars")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--catalog", default=None, help="path to cached DR3 CSV")
    p.add_argument("--n_draws", type=int, default=None)
    p.add_argument("--max_stars", type=int, default=None)
    p.add_argument("--apertures", type=float, nargs="+", default=None)
    # observation mode (detection = broadband; characterization = spectroscopy)
    p.add_argument("--mode", choices=["detection", "characterization"],
                   default="detection")
    p.add_argument("--spectral_R", type=float, default=None,
                   help="spectral resolution for characterization (default 70)")
    p.add_argument("--snr_detect", type=float, default=None,
                   help="broadband detection SNR threshold (default 10)")
    p.add_argument("--snr_char", type=float, default=None,
                   help="per-element characterization SNR threshold (default 10)")
    p.add_argument("--exp_detect_min", type=float, default=None,
                   help="detection integration per target [minutes] (default 5)")
    p.add_argument("--exp_char_min", type=float, default=None,
                   help="characterization integration per target [minutes] (default 360)")
    p.add_argument("--eta_earth", type=float, default=None)
    p.add_argument("--k_owa", type=float, default=None,
                   help="enable outer working angle = k_owa * lambda/D")
    p.add_argument("--bootstrap", type=int, default=0,
                   help="bootstrap resamples of the stellar sample for CIs (0=off)")
    p.add_argument("--no_plots", action="store_true")
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = build_config(args)

    result = pipeline.run(cfg, base_dir=base_dir, test_mode=args.test,
                          n_bootstrap=args.bootstrap)
    out_dir = os.path.join(base_dir, args.out_dir)
    pipeline.save(result, out_dir)
    print(f"\n>> results written to {out_dir}")

    if not args.no_plots:
        import plots
        fig_dir = os.path.join(out_dir, "figures")
        plots.make_all(result, cfg, fig_dir)
        print(f">> figures written to {fig_dir}")


if __name__ == "__main__":
    main()
