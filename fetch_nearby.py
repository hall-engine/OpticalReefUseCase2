#!/usr/bin/env python3
"""
fetch_nearby.py  --  COMPLETE nearby-star census for the mission comparison.

Unlike the volume-limited pole sample, this grabs EVERY FGK dwarf inside a small
distance (default 1-40 pc) -- no random subsampling -- so the yield is a direct
count over the local population that HWO/LUVOIR-class telescopes actually target.
40 pc covers the 15 m LUVOIR-A reach (its 23 mas IWA can resolve an HZ Earth out
to ~44 pc). The volume is tiny (parallax > 25 mas), so ONE TAP query suffices.

Writes gaia_nearby.csv (+ .meta.json with rand_fraction = 1.0 -> no extrapolation).

    python3 fetch_nearby.py                 # 1-40 pc, complete
    python3 fetch_nearby.py --dist_max_pc 30
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

import pandas as pd
import requests

SERVERS = [
    ("ESA archive", "https://gea.esac.esa.int/tap-server/tap/sync"),
    ("ARI mirror",  "https://gaia.ari.uni-heidelberg.de/tap/sync"),
]


def query(dist_min_pc, dist_max_pc, poe_min, ruwe_max=1.4):
    parallax_max = 1000.0 / dist_min_pc
    parallax_min = 1000.0 / dist_max_pc
    return (
        "SELECT g.source_id, g.ra, g.dec, g.parallax, g.parallax_over_error, "
        "g.phot_g_mean_mag, g.random_index, g.ruwe, "
        "ap.teff_gspphot, ap.logg_gspphot, ap.lum_flame, "
        "ap.mass_flame, ap.radius_flame "
        "FROM gaiadr3.gaia_source AS g "
        "JOIN gaiadr3.astrophysical_parameters AS ap ON g.source_id = ap.source_id "
        f"WHERE g.parallax BETWEEN {parallax_min} AND {parallax_max} "
        f"AND g.parallax_over_error > {poe_min} "
        # astrometric-quality cut (RUWE): drop likely unresolved binaries, whose
        # blended light corrupts lum_flame and which are poor imaging targets.
        f"AND g.ruwe < {ruwe_max} "
        "AND ap.lum_flame IS NOT NULL "
        "AND ap.teff_gspphot IS NOT NULL "
        "AND ap.logg_gspphot IS NOT NULL"
    )


def fetch(url, q, timeout):
    r = requests.get(url, params={"REQUEST": "doQuery", "LANG": "ADQL",
                                  "FORMAT": "csv", "QUERY": q,
                                  "MAXREC": "500000"}, timeout=timeout)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dist_min_pc", type=float, default=1.0)
    p.add_argument("--dist_max_pc", type=float, default=40.0)
    p.add_argument("--poe_min", type=float, default=5.0)
    p.add_argument("--ruwe_max", type=float, default=1.4,
                   help="reject RUWE >= this (likely unresolved binaries / bad "
                        "astrometry). 1.4 is the Gaia-standard cut; 99 disables.")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--out", default="gaia_nearby.csv")
    args = p.parse_args()

    q = query(args.dist_min_pc, args.dist_max_pc, args.poe_min, args.ruwe_max)
    print(f">> nearby census {args.dist_min_pc}-{args.dist_max_pc} pc "
          f"(parallax {1000/args.dist_max_pc:.1f}-{1000/args.dist_min_pc:.0f} mas), "
          f"complete (no subsampling)", flush=True)

    df = None
    for attempt in range(1, args.retries + 1):
        for name, url in SERVERS:
            try:
                print(f">> attempt {attempt}: {name} ...", flush=True)
                t0 = time.time()
                df = fetch(url, q, args.timeout)
                print(f"   got {len(df)} rows in {time.time()-t0:.0f}s from {name}",
                      flush=True)
                break
            except Exception as exc:                    # noqa: BLE001
                print(f"   {name} failed: {type(exc).__name__}: {str(exc)[:90]}",
                      flush=True)
        if df is not None:
            break
        time.sleep(min(5 * attempt, 20))

    if df is None:
        sys.exit(">> no TAP server responded; try again later.")

    df = df.drop_duplicates("source_id")
    df.to_csv(args.out, index=False)
    with open(os.path.splitext(args.out)[0] + ".meta.json", "w") as fh:
        json.dump({"rand_fraction": 1.0,              # complete census
                   "dist_min_pc": args.dist_min_pc,
                   "dist_max_pc": args.dist_max_pc,
                   "poe_min": args.poe_min,
                   "ruwe_max": args.ruwe_max,
                   "n_rows": int(len(df))}, fh, indent=2)
    print(f">> DONE: {len(df)} nearby stars -> {args.out} (+ .meta.json)", flush=True)


if __name__ == "__main__":
    main()
