#!/usr/bin/env python3
"""
fetch_local.py  --  one-time Gaia DR3 fetch via TAP *sync*, chunked + PARALLEL.

The gaia_source x astrophysical_parameters join over a large random_index range
times out as one query, so we split it into small random_index chunks (~1000 rows
/ ~9 s each) and fetch them CONCURRENTLY with a thread pool (each chunk is an
independent HTTP request that mostly just waits on the server). Plain sync HTTP
per chunk -- no astroquery/pyvo/async. Tries ESA first, then the ARI mirror.

Writes gaia_pole_sample.csv (+ .meta.json) in the pipeline's format.

    python3 fetch_local.py --rand_fraction 0.0005            # tiny test
    python3 fetch_local.py --rand_fraction 0.1 --workers 12  # full analysis
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def fetch_one(url, lo, hi, args):
    """Fetch a single chunk with a few quick retries. Returns a DataFrame."""
    q = chunk_query(lo, hi, args.dist_min_pc, args.dist_max_pc)
    last = None
    for attempt in range(1, args.retries + 1):
        try:
            return fetch_chunk(url, q, args.chunk_timeout)
        except Exception as exc:                        # noqa: BLE001
            last = exc
            time.sleep(min(2 * attempt, 8))
    raise RuntimeError(f"chunk {lo}-{hi}: {type(last).__name__}: {str(last)[:80]}")


def probe(lo, hi, args):
    """Try servers in order on one chunk; return (name, url) of the first that works."""
    q = chunk_query(lo, hi, args.dist_min_pc, args.dist_max_pc)
    for name, url in SERVERS:
        print(f">> probing {name} (up to {args.probe_timeout:.0f}s) ...", flush=True)
        try:
            fetch_chunk(url, q, args.probe_timeout)
            print(f"   using {name}", flush=True)
            return name, url
        except Exception as exc:                        # noqa: BLE001
            print(f"   {name} unavailable: {type(exc).__name__}: "
                  f"{str(exc)[:80]}", flush=True)
    return None, None


def fetch_ranges(url, ranges, args, label=""):
    """Fetch many chunks concurrently. Returns (results dict, failed list)."""
    results, failed = {}, []
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, url, lo, hi, args): (lo, hi) for lo, hi in ranges}
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                results[key] = fut.result()
            except Exception:                           # noqa: BLE001
                failed.append(key)
            done += 1
            if done % args.workers == 0 or done == len(ranges):
                rows = sum(len(d) for d in results.values())
                print(f"   {label}{done}/{len(ranges)} chunks, {rows:,} rows, "
                      f"{len(failed)} failed, {time.time()-t0:.0f}s", flush=True)
    return results, failed


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rand_fraction", type=float, default=0.1)
    p.add_argument("--dist_min_pc", type=float, default=10.0)
    p.add_argument("--dist_max_pc", type=float, default=300.0)
    p.add_argument("--chunk_size", type=int, default=540_000,
                   help="random_index span per chunk (~955 rows/~9s at this value)")
    p.add_argument("--workers", type=int, default=12,
                   help="concurrent chunk requests (lower if the server rate-limits)")
    p.add_argument("--chunk_timeout", type=float, default=45.0,
                   help="per-request timeout; a stall fails fast instead of hanging")
    p.add_argument("--probe_timeout", type=float, default=15.0)
    p.add_argument("--retries", type=int, default=4, help="attempts per chunk")
    p.add_argument("--fill_rounds", type=int, default=4,
                   help="extra passes to re-fetch any chunks that failed")
    p.add_argument("--probe_wait", type=float, default=30.0,
                   help="wait between fill rounds if the server is momentarily down")
    p.add_argument("--out", default="gaia_pole_sample.csv")
    args = p.parse_args()

    rand_end = int(args.rand_fraction * RAND_MAX)
    edges = list(range(0, rand_end, args.chunk_size)) + [rand_end]
    ranges = [(edges[i], edges[i + 1] - 1) for i in range(len(edges) - 1)]
    print(f">> fetch rand_fraction={args.rand_fraction} "
          f"({args.dist_min_pc}-{args.dist_max_pc} pc): {len(ranges)} chunks x "
          f"{args.chunk_size:,}, {args.workers} parallel workers", flush=True)

    server, url = probe(ranges[0][0], ranges[0][1], args)
    if url is None:
        sys.exit(">> no TAP server responded; try again later.")

    t0 = time.time()
    results, failed = fetch_ranges(url, ranges, args)

    # fill rounds for any stragglers (re-probe each round in case a server blipped)
    for rnd in range(1, args.fill_rounds + 1):
        if not failed:
            break
        print(f">> fill round {rnd}/{args.fill_rounds}: {len(failed)} chunks left; "
              f"re-probing ...", flush=True)
        _, u2 = probe(failed[0][0], failed[0][1], args)
        if u2 is None:
            time.sleep(args.probe_wait)
            continue
        more, failed = fetch_ranges(u2, failed, args, label=f"fill{rnd} ")
        results.update(more)

    if not results:
        sys.exit(">> no chunks fetched at all; servers unreachable, rerun later.")

    # always write what we have -- disjoint chunks stay a valid random subset
    cat = pd.concat(results.values(), ignore_index=True).drop_duplicates("source_id")
    cat.to_csv(args.out, index=False)
    got = len(ranges) - len(failed)
    with open(os.path.splitext(args.out)[0] + ".meta.json", "w") as fh:
        json.dump({"rand_fraction": args.rand_fraction,
                   "dist_min_pc": args.dist_min_pc,
                   "dist_max_pc": args.dist_max_pc,
                   "n_rows": int(len(cat)),
                   "n_fetch_chunks": got,
                   "n_chunks_planned": len(ranges),
                   "n_chunks_missing": len(failed)}, fh, indent=2)
    print(f">> DONE: {len(cat):,} rows from {got}/{len(ranges)} chunks -> "
          f"{args.out} (+ .meta.json) in {time.time()-t0:.0f}s", flush=True)
    if failed:
        print(f">> NOTE: {len(failed)} chunks never succeeded; the sample is "
              f"slightly smaller but still an unbiased random subset. Rerun to "
              f"top up if you want the full fraction.", flush=True)


if __name__ == "__main__":
    main()
