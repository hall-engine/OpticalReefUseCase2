#!/usr/bin/env python3
"""
run_cube.py
===========
Compute the 4-D design-trade datacube for the Optical Reef yield analysis:

        aperture D  x  contrast floor  x  integration time  x  k_OWA

For every cell it stores the ACTUAL completeness (fraction of viable candidates
that yield a detectable HZ Earth analog) plus the three limit fractions (IWA,
OWA, contrast/SNR) so EVERY figure can be rebuilt locally with cube_figures.py
without ever re-running the Monte Carlo.

Design points that make this HPC-friendly:
  * geometry is sampled ONCE per star (aperture/contrast/exposure/k_OWA free);
  * stars are processed in CHUNKS so memory stays bounded at full DR3 scale;
  * counts are additive over stars, so an array job can shard the star list and
    the shards are summed by `--combine`.

Usage
-----
Single node (whole selected sample, memory-safe via chunking):
    python run_cube.py --out cube.npz

Array job (shard i of N; each writes a partial-count file):
    python run_cube.py --shard $SLURM_ARRAY_TASK_ID --nshards $N --tmp parts/
Then combine:
    python run_cube.py --combine parts/ --out cube.npz

Axes (all CLI, nothing hardcoded):
    --apertures --contrasts --times_min --kowa --mode --spectral_R --snr
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import socket
import subprocess
import sys

import numpy as np

from config import RunConfig
import catalog
import montecarlo
import photometry

try:
    import h5py
    HAVE_H5 = True
except Exception:
    HAVE_H5 = False

MIN = 60.0
AXIS_KEYS = ("apertures_m", "contrasts", "times_min", "kowa")


def _git_commit(base_dir):
    try:
        return subprocess.check_output(
            ["git", "-C", base_dir, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
def _axes(args):
    apertures = (np.array(args.apertures, float) if args.apertures
                 else np.geomspace(args.dmin, args.dmax, args.n_apertures))
    contrasts = np.array(args.contrasts, float)
    times_min = np.array(args.times_min, float)
    kowa = np.array(args.kowa, float)
    return apertures, contrasts, times_min, kowa


def _accumulate(cfg, args, apertures, contrasts, times_min, kowa, base_dir):
    """Return summed detection counts over the (sharded) star sample plus the
    number of (star x draw) trials, all as integer counts."""
    rng = np.random.default_rng(cfg.montecarlo.seed + 1000 * args.shard)
    raw = catalog.load_catalog(cfg.survey, base_dir=base_dir)
    stars, n_full = catalog.select_stars(raw, cfg.survey, cfg.montecarlo,
                                         test_mode=bool(args.max_stars), rng=rng)

    # shard the star list (strided so each shard is representative)
    if args.nshards > 1:
        stars = stars.iloc[args.shard::args.nshards].reset_index(drop=True)

    n_sim = len(stars)
    ndraws = cfg.montecarlo.n_draws
    A, C, E, K = len(apertures), len(contrasts), len(times_min), len(kowa)

    det_count = np.zeros((A, C, E, K), dtype=np.int64)
    iwa_count = np.zeros(A, dtype=np.int64)
    owa_count = np.zeros((A, K), dtype=np.int64)
    snr_count = np.zeros((A, C, E), dtype=np.int64)

    inst = cfg.instrument
    bw = (1.0 / args.spectral_R) if args.mode == "characterization" else inst.bandwidth
    lam = inst.lambda0_m

    g_all = stars["phot_g_mean_mag"].to_numpy()
    lum_all = stars["lum_flame"].to_numpy()
    hzi = stars["hz_inner_m"].to_numpy()
    hzo = stars["hz_outer_m"].to_numpy()
    dist = stars["distance_m"].to_numpy()

    batch = args.star_batch
    n_trials = 0
    for b0 in range(0, n_sim, batch):
        b1 = min(b0 + batch, n_sim)
        orb = montecarlo.sample_orbits(hzi[b0:b1], hzo[b0:b1], dist[b0:b1],
                                       cfg.montecarlo, rng=rng)
        ang = orb.ang_sep_rad                       # (nb, ndraws)
        g = g_all[b0:b1]
        lum = lum_all[b0:b1]
        n_trials += (b1 - b0) * ndraws

        for ai, D in enumerate(apertures):
            iwa = inst.k_iwa * lam / D
            iwa_ok = ang > iwa
            iwa_count[ai] += int(iwa_ok.sum())
            owa_ok = []
            for ki, k in enumerate(kowa):
                ok = np.ones_like(iwa_ok) if k <= 0 else (ang < (k * lam / D))
                owa_ok.append(ok)
                owa_count[ai, ki] += int(ok.sum())
            for ci, con in enumerate(contrasts):
                for ei, t in enumerate(times_min):
                    det = photometry.detectability(
                        orb, g, lum, D, con, t * MIN, inst,
                        bandwidth_frac=bw, target_snr=args.snr)
                    snr_ok = det["snr_ok"]
                    snr_count[ai, ci, ei] += int(snr_ok.sum())
                    geo_snr = iwa_ok & snr_ok
                    for ki in range(K):
                        det_count[ai, ci, ei, ki] += int(
                            (geo_snr & owa_ok[ki]).sum())
        print(f"   stars {b1}/{n_sim}", end="\r")
    print()
    return dict(det_count=det_count, iwa_count=iwa_count, owa_count=owa_count,
                snr_count=snr_count, n_trials=np.int64(n_trials),
                n_sim=np.int64(n_sim), n_full=np.int64(n_full))


def _meta(cfg, args, apertures, contrasts, times_min, kowa, base_dir):
    """Axes + model parameters + full run provenance, so the cube is
    self-describing and reproducible long after the run."""
    return dict(
        # --- axes ---
        apertures_m=apertures, contrasts=contrasts, times_min=times_min,
        kowa=kowa,
        # --- model ---
        n_draws=np.int64(cfg.montecarlo.n_draws),
        seed=np.int64(cfg.montecarlo.seed),
        rand_fraction=np.float64(cfg.survey.rand_fraction or 1.0),
        eta_earth=np.float64(cfg.drake.eta_earth),
        dist_min_pc=np.float64(cfg.survey.dist_min_pc),
        dist_max_pc=np.float64(cfg.survey.dist_max_pc),
        k_iwa=np.float64(cfg.instrument.k_iwa),
        lambda0_m=np.float64(cfg.instrument.lambda0_m),
        bandwidth=np.float64(cfg.instrument.bandwidth),
        eta_inst=np.float64(cfg.instrument.eta_inst),
        eta_coron=np.float64(cfg.instrument.eta_coron),
        sigma_sys=np.float64(cfg.instrument.sigma_sys),
        nzodi_level=np.float64(cfg.instrument.nzodi_level),
        snr_threshold=np.float64(args.snr),
        mode=args.mode, spectral_R=np.float64(args.spectral_R),
        hz_inner_au=np.float64(cfg.montecarlo.hz_inner_au),
        hz_outer_au=np.float64(cfg.montecarlo.hz_outer_au),
        geometric_albedo=np.float64(cfg.montecarlo.geometric_albedo),
        # --- provenance ---
        created_utc=datetime.datetime.utcnow().isoformat() + "Z",
        git_commit=_git_commit(base_dir),
        command=" ".join(sys.argv),
        hostname=socket.gethostname(),
        catalog_path=cfg.survey.catalog_path,
        config_json=json.dumps(cfg.to_dict()),
        dims="completeness[aperture, contrast, time, kowa]",
    )


def _attr_value(v):
    """Coerce a metadata value to something h5py accepts as an attribute
    (python scalar/str), including numpy 0-d arrays loaded back from npz shards."""
    if isinstance(v, np.ndarray):
        return v.item() if v.ndim == 0 else v
    if isinstance(v, np.generic):
        return v.item()
    return v


def _resolve_format(out_path, fmt):
    ext = os.path.splitext(out_path)[1].lower()
    if ext in (".h5", ".hdf5"):
        chosen = "h5"
    elif ext == ".npz":
        chosen = "npz"
    else:
        chosen = ("h5" if HAVE_H5 else "npz") if fmt == "auto" else fmt
        out_path = out_path + (".h5" if chosen == "h5" else ".npz")
    if chosen == "h5" and not HAVE_H5:
        print(">> h5py not available; falling back to .npz")
        chosen = "npz"
        out_path = os.path.splitext(out_path)[0] + ".npz"
    return out_path, chosen


def _finalize(counts, meta, out_path, fmt="auto"):
    """Turn summed counts into fractions and write the self-describing cube."""
    out_path, chosen = _resolve_format(out_path, fmt)
    denom = float(counts["n_trials"])
    completeness = (counts["det_count"] / denom).astype(np.float32)  # (A,C,E,K)
    frac_iwa = (counts["iwa_count"] / denom).astype(np.float32)      # (A,)
    frac_owa = (counts["owa_count"] / denom).astype(np.float32)      # (A,K)
    frac_snr = (counts["snr_count"] / denom).astype(np.float32)      # (A,C,E)
    scalars = {k: v for k, v in meta.items() if k not in AXIS_KEYS}
    scalars.update(n_sim=int(counts["n_sim"]), n_full=int(counts["n_full"]),
                   n_trials=int(counts["n_trials"]))
    axes = {k: np.asarray(meta[k]) for k in AXIS_KEYS}

    if chosen == "h5":
        with h5py.File(out_path, "w") as f:
            gax = f.create_group("axes")
            for k, v in axes.items():
                gax.create_dataset(k, data=v)
            f.create_dataset("completeness", data=completeness,
                             compression="gzip")
            f.create_dataset("frac_iwa", data=frac_iwa)
            f.create_dataset("frac_owa", data=frac_owa)
            f.create_dataset("frac_snr", data=frac_snr)
            f["completeness"].attrs["dims"] = "aperture,contrast,time,kowa"
            for k, v in scalars.items():
                f.attrs[k] = _attr_value(v)   # metadata -> file attributes
    else:
        np.savez_compressed(out_path, **axes,
                            completeness=completeness, frac_iwa=frac_iwa,
                            frac_owa=frac_owa, frac_snr=frac_snr, **scalars)

    # human-readable JSON sidecar (both formats)
    side = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in {**axes, **scalars}.items()}
    with open(os.path.splitext(out_path)[0] + ".json", "w") as fh:
        json.dump(side, fh, indent=2, default=str)
    print(f">> wrote {out_path}  (completeness {completeness.shape}, "
          f"n_full={int(counts['n_full'])}, format={chosen})")


# --------------------------------------------------------------------------
def combine(parts_dir, out_path, fmt="auto"):
    files = sorted(glob.glob(os.path.join(parts_dir, "part_*.npz")))
    if not files:
        raise SystemExit(f"no part_*.npz in {parts_dir}")
    print(f">> combining {len(files)} shards")
    acc = None
    meta = None
    for f in files:
        d = np.load(f, allow_pickle=True)
        if acc is None:
            acc = {k: d[k].copy() for k in
                   ("det_count", "iwa_count", "owa_count", "snr_count")}
            n_trials = d["n_trials"].copy()
            n_sim = d["n_sim"].copy()
            n_full = d["n_full"].copy()
            meta = {k: d[k] for k in d.files
                    if k not in ("det_count", "iwa_count", "owa_count",
                                 "snr_count", "n_trials", "n_sim", "n_full")}
        else:
            for k in acc:
                acc[k] += d[k]
            n_trials += d["n_trials"]
            n_sim += d["n_sim"]
            # n_full identical across shards (full selection); keep first
    acc.update(n_trials=n_trials, n_sim=n_sim, n_full=n_full)
    meta = {k: meta[k] for k in meta}
    meta["combined_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
    meta["n_shards"] = len(files)
    _finalize(acc, meta, out_path, fmt)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="cube.h5")
    p.add_argument("--format", choices=["auto", "h5", "npz"], default="auto",
                   help="cube format; auto = h5 if h5py present else npz")
    p.add_argument("--combine", default=None,
                   help="directory of part_*.npz shards to merge, then exit")
    # axes
    p.add_argument("--apertures", type=float, nargs="+", default=None)
    p.add_argument("--n_apertures", type=int, default=30)
    p.add_argument("--dmin", type=float, default=4.0)
    p.add_argument("--dmax", type=float, default=1000.0)
    p.add_argument("--contrasts", type=float, nargs="+",
                   default=[1e-9, 3e-10, 1e-10, 3e-11, 1e-11])
    p.add_argument("--times_min", type=float, nargs="+",
                   default=[5, 15, 30, 60, 120, 240, 480, 720])
    p.add_argument("--kowa", type=float, nargs="+",
                   default=[8, 16, 32, 64, 128, 0],   # 0 = no OWA limit
                   help="k_OWA values (OWA=k*lambda/D); 0 disables the limit")
    # model
    p.add_argument("--mode", choices=["detection", "characterization"],
                   default="detection")
    p.add_argument("--spectral_R", type=float, default=70.0)
    p.add_argument("--snr", type=float, default=10.0)
    p.add_argument("--n_draws", type=int, default=2000)
    p.add_argument("--max_stars", type=int, default=None,
                   help="cap simulated stars (random subsample); default full")
    p.add_argument("--star_batch", type=int, default=20000)
    p.add_argument("--test", action="store_true",
                   help="quick smoke test: 500 stars, 150 draws, coarse grid "
                        "(verifies env/catalog/output in seconds)")
    # sharding for array jobs
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--tmp", default="parts",
                   help="dir for partial-count files when nshards>1")
    p.add_argument("--catalog", default=None)
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.combine:
        combine(os.path.join(base_dir, args.combine)
                if not os.path.isabs(args.combine) else args.combine,
                os.path.join(base_dir, args.out) if not os.path.isabs(args.out)
                else args.out, fmt=args.format)
        return

    if args.test:                                      # quick smoke test
        args.max_stars = args.max_stars or 500
        args.n_draws = 150
        if args.apertures is None:
            args.n_apertures = min(args.n_apertures, 12)

    cfg = RunConfig()
    cfg.montecarlo.n_draws = args.n_draws
    if args.catalog:
        cfg.survey.catalog_path = args.catalog
    if args.max_stars:
        cfg.survey.test_max_stars = args.max_stars
    cfg.instrument.k_owa = None                        # k_OWA is a cube axis here

    apertures, contrasts, times_min, kowa = _axes(args)
    print(f">> cube axes: {len(apertures)} D x {len(contrasts)} contrast x "
          f"{len(times_min)} time x {len(kowa)} kOWA  | mode={args.mode} "
          f"| draws={args.n_draws} | shard {args.shard}/{args.nshards}")

    counts = _accumulate(cfg, args, apertures, contrasts, times_min, kowa, base_dir)
    meta = _meta(cfg, args, apertures, contrasts, times_min, kowa, base_dir)

    if args.nshards > 1:
        tmp = os.path.join(base_dir, args.tmp) if not os.path.isabs(args.tmp) else args.tmp
        os.makedirs(tmp, exist_ok=True)
        part = os.path.join(tmp, f"part_{args.shard:04d}.npz")
        np.savez_compressed(part, **counts, **meta)
        print(f">> wrote shard {part}")
    else:
        out = os.path.join(base_dir, args.out) if not os.path.isabs(args.out) else args.out
        _finalize(counts, meta, out, fmt=args.format)


if __name__ == "__main__":
    main()
