#!/usr/bin/env python3
"""
fetch_local.py  --  one-time Gaia DR3 fetch via TAP *sync*, in CHUNKS.

Why chunks: the gaia_source x astrophysical_parameters join over a large
random_index range is expensive and times out as one big query (this is the bug
that broke us). Splitting the random_index range into many small chunks makes
each query cheap and fast (~1000 rows / ~10 s), then we concatenate. This is what
the original working catalog did (its meta says n_fetch_chunks: 100).

Plain synchronous HTTP per chunk (requests only -- no astroquery/pyvo/async).
Tries the ESA archive first, falls back to the ARI-Heidelberg mirror. Writes
gaia_pole_sample.csv (+ .meta.json) in the pipeline's format.

    python3 fetch_local.py --rand_fraction 0.0005   # tiny test
    python3 fetch_local.py --rand_fraction 0.1      # full analysis
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

RAND_MAX = 1_800_000_000
SERVERS = [
    ("ESA archive", "https://gea.esac.esa.int/tap-server/tap/sync"),
    ("ARI mirror",  "https://gaia.ari.uni-heidelberg.de/tap/sync"),
]


def chunk_query(lo, hi, dist_min_pc, dist_max_pc):
    parallax_max = 1000.0 / dist_min_pc
    parallax_min = 1000.0 / dist_max_pc
    return (
        "SELECT g.source_id, g.ra, g.dec, g.parallax, g.parallax_over_error, "
        "g.phot_g_mean_mag, g.random_index, "
        "ap.teff_gspphot, ap.logg_gspphot, ap.lum_flame, "
        "ap.mass_flame, ap.radius_flame "
        "FROM gaiadr3.gaia_source AS g "
        "JOIN gaiadr3.astrophysical_parameters AS ap ON g.source_id = ap.source_id "
        f"WHERE g.parallax BETWEEN {parallax_min} AND {parallax_max} "
        "AND g.parallax_over_error > 5 "
        "AND ap.lum_flame IS NOT NULL "
        "AND ap.teff_gspphot IS NOT NULL "
        "AND ap.logg_gspphot IS NOT NULL "
        f"AND g.random_index BETWEEN {lo} AND {hi}"
    )


def fetch_chunk(url, query, timeout):
    r = requests.get(url, params={"REQUEST": "doQuery", "LANG": "ADQL",
                                  "FORMAT": "csv", "QUERY": query,
                                  "MAXREC": "1000000"}, timeout=timeout)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def get_chunk(url, lo, hi, args):
    """Fetch one chunk from a chosen server, retrying transient errors."""
    q = chunk_query(lo, hi, args.dist_min_pc, args.dist_max_pc)
    last = None
    for attempt in range(1, args.retries + 1):
        try:
            return fetch_chunk(url, q, args.chunk_timeout)
        except Exception as exc:                        # noqa: BLE001
            last = exc
            print(f"     attempt {attempt}/{args.retries} failed: "
                  f"{type(exc).__name__}: {str(exc)[:90]}", flush=True)
            time.sleep(min(5 * attempt, 20))
    raise RuntimeError(f"chunk {lo}-{hi} failed: {last}")


def probe(lo, hi, args):
    """Try the servers in order on the (lo, hi) chunk; return (name, url, df) for
    the first that works. Used at the start and to re-select a live server if the
    current one dies mid-run (so an unattended job self-heals)."""
    q = chunk_query(lo, hi, args.dist_min_pc, args.dist_max_pc)
    for name, url in SERVERS:
        print(f">> probing {name} (up to {args.probe_timeout:.0f}s) ...", flush=True)
        try:
            df = fetch_chunk(url, q, args.probe_timeout)
            print(f"   using {name} ({len(df)} rows)", flush=True)
            return name, url, df
        except Exception as exc:                        # noqa: BLE001
            print(f"   {name} unavailable: {type(exc).__name__}: "
                  f"{str(exc)[:90]}", flush=True)
    return None, None, None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rand_fraction", type=float, default=0.1)
    p.add_argument("--dist_min_pc", type=float, default=10.0)
    p.add_argument("--dist_max_pc", type=float, default=300.0)
    p.add_argument("--chunk_size", type=int, default=540_000,
                   help="random_index span per chunk (~955 rows/~9s at this value)")
    p.add_argument("--chunk_timeout", type=float, default=120.0)
    p.add_argument("--probe_timeout", type=float, default=45.0,
                   help="timeout for the initial server probe (dead ESA fails fast)")
    p.add_argument("--retries", type=int, default=5, help="attempts per chunk on a server")
    p.add_argument("--probe_retries", type=int, default=30,
                   help="times to re-probe (waiting) if BOTH servers are down")
    p.add_argument("--probe_wait", type=float, default=120.0,
                   help="seconds between probe rounds when both servers are down")
    p.add_argument("--out", default="gaia_pole_sample.csv")
    args = p.parse_args()

    rand_end = int(args.rand_fraction * RAND_MAX)
    edges = list(range(0, rand_end, args.chunk_size)) + [rand_end]
    n_chunks = len(edges) - 1
    print(f">> fetch rand_fraction={args.rand_fraction} "
          f"({args.dist_min_pc}-{args.dist_max_pc} pc) in {n_chunks} chunks "
          f"of {args.chunk_size:,} random_index each", flush=True)

    def probe_with_wait(lo, hi):
        """Probe both servers, waiting/retrying if both are down (unattended-safe)."""
        for r in range(1, args.probe_retries + 1):
            name, url, df = probe(lo, hi, args)
            if url is not None:
                return name, url, df
            print(f">> both servers down (probe round {r}/{args.probe_retries}); "
                  f"waiting {args.probe_wait:.0f}s ...", flush=True)
            time.sleep(args.probe_wait)
        return None, None, None

    t0 = time.time()
    server, url, df0 = probe_with_wait(edges[0], edges[1] - 1)
    if url is None:
        sys.exit(">> no TAP server responded after all probe rounds; rerun later.")
    parts = [df0]
    print(f"   chunk 1/{n_chunks} [{server}]: {len(df0)} rows", flush=True)

    for i in range(1, n_chunks):
        lo, hi = edges[i], edges[i + 1] - 1
        try:
            df = get_chunk(url, lo, hi, args)
        except Exception as exc:                        # chosen server died mid-run
            print(f"   chunk {i+1} failed on {server} ({str(exc)[:80]}); "
                  f"re-probing servers ...", flush=True)
            server, url, df = probe_with_wait(lo, hi)    # returns this chunk's df too
            if url is None:
                sys.exit(f">> all servers down at chunk {i+1}; rerun later "
                         f"(nothing saved yet).")
        parts.append(df)
        total = sum(len(d) for d in parts)
        print(f"   chunk {i+1}/{n_chunks} [{server}]: {len(df)} rows "
              f"(cumulative {total:,}, {time.time()-t0:.0f}s elapsed)", flush=True)

    cat = pd.concat(parts, ignore_index=True).drop_duplicates("source_id")
    cat.to_csv(args.out, index=False)
    with open(os.path.splitext(args.out)[0] + ".meta.json", "w") as fh:
        json.dump({"rand_fraction": args.rand_fraction,
                   "dist_min_pc": args.dist_min_pc,
                   "dist_max_pc": args.dist_max_pc,
                   "n_rows": int(len(cat)),
                   "n_fetch_chunks": n_chunks}, fh, indent=2)
    print(f">> DONE: {len(cat):,} rows -> {args.out} (+ .meta.json) "
          f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
