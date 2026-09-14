#!/usr/bin/env python3
"""
fetch_galaxy_context.py  --  tiny all-sky Gaia DR3 sample for a GALACTIC-CONTEXT
figure (not the yield analysis).

Pulls only gaia_source (ra, dec, parallax, G) -- no astrophysical_parameters
join -- so it is fast. A very small random_index fraction across a WIDE distance
range (default 10 pc .. 5 kpc) is enough to trace the thin disk edge-on and place
the 300 pc habitable-survey sphere in context. Chunked + parallel sync TAP, ESA
first then the ARI mirror (same robust pattern as fetch_local.py).

Writes gaia_galaxy_context.csv (+ .meta.json).

    python3 fetch_galaxy_context.py --rand_fraction 0.0005      # ~tiny, quick
    python3 fetch_galaxy_context.py --rand_fraction 0.002 --workers 4 --delay 1.5
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

RAND_MAX = 1_800_000_000
SERVERS = [
    ("ESA archive", "https://gea.esac.esa.int/tap-server/tap/sync"),
    ("ARI mirror",  "https://gaia.ari.uni-heidelberg.de/tap/sync"),
]


class RateLimiter:
    """Minimum interval between request STARTS across all threads (per-IP pace)."""
    def __init__(self, min_interval):
        self.min_interval = float(min_interval)
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self):
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.min_interval
        gap = start - time.monotonic()
        if gap > 0:
            time.sleep(gap)


_LIMITER = None


def chunk_query(lo, hi, dist_min_pc, dist_max_pc, poe_min, gmax):
    parallax_max = 1000.0 / dist_min_pc
    parallax_min = 1000.0 / dist_max_pc
    return (
        "SELECT g.source_id, g.ra, g.dec, g.parallax, g.parallax_over_error, "
        "g.phot_g_mean_mag, g.random_index "
        "FROM gaiadr3.gaia_source AS g "
        f"WHERE g.parallax BETWEEN {parallax_min} AND {parallax_max} "
        f"AND g.parallax_over_error > {poe_min} "
        f"AND g.phot_g_mean_mag < {gmax} "
        f"AND g.random_index BETWEEN {lo} AND {hi}"
    )


def fetch_chunk(url, query, timeout):
    if _LIMITER is not None:
        _LIMITER.wait()
    r = requests.get(url, params={"REQUEST": "doQuery", "LANG": "ADQL",
                                  "FORMAT": "csv", "QUERY": query,
                                  "MAXREC": "3000000"}, timeout=timeout)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def fetch_one(url, lo, hi, args):
    q = chunk_query(lo, hi, args.dist_min_pc, args.dist_max_pc,
                    args.poe_min, args.gmax)
    last = None
    for attempt in range(1, args.retries + 1):
        try:
            return fetch_chunk(url, q, args.chunk_timeout)
        except Exception as exc:                        # noqa: BLE001
            last = exc
            time.sleep(min(2 * attempt, 8))
    raise RuntimeError(f"chunk {lo}-{hi}: {type(last).__name__}: {str(last)[:80]}")


def probe(lo, hi, args):
    q = chunk_query(lo, hi, args.dist_min_pc, args.dist_max_pc,
                    args.poe_min, args.gmax)
    for name, url in SERVERS:
        print(f">> probing {name} (up to {args.probe_timeout:.0f}s) ...", flush=True)
        try:
            df = fetch_chunk(url, q, args.probe_timeout)
            print(f"   using {name} ({len(df)} rows in probe chunk)", flush=True)
            return name, url
        except Exception as exc:                        # noqa: BLE001
            print(f"   {name} unavailable: {type(exc).__name__}: "
                  f"{str(exc)[:80]}", flush=True)
    return None, None


def fetch_ranges(url, ranges, args, label=""):
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
            if done % max(1, args.workers) == 0 or done == len(ranges):
                rows = sum(len(d) for d in results.values())
                print(f"   {label}{done}/{len(ranges)} chunks, {rows:,} rows, "
                      f"{len(failed)} failed, {time.time()-t0:.0f}s", flush=True)
    return results, failed


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rand_fraction", type=float, default=0.0008)
    p.add_argument("--dist_min_pc", type=float, default=10.0)
    p.add_argument("--dist_max_pc", type=float, default=5000.0)
    p.add_argument("--poe_min", type=float, default=5.0,
                   help="min parallax_over_error (distance reliability)")
    p.add_argument("--gmax", type=float, default=17.0,
                   help="faint G limit; brighter stars have better parallaxes")
    p.add_argument("--chunk_size", type=int, default=300_000,
                   help="random_index span per chunk")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--delay", type=float, default=1.0,
                   help="min seconds between request STARTS (per-IP pacing)")
    p.add_argument("--chunk_timeout", type=float, default=60.0)
    p.add_argument("--probe_timeout", type=float, default=20.0)
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--fill_rounds", type=int, default=4)
    p.add_argument("--probe_wait", type=float, default=30.0)
    p.add_argument("--out", default="gaia_galaxy_context.csv")
    args = p.parse_args()

    global _LIMITER
    if args.delay > 0:
        _LIMITER = RateLimiter(args.delay)

    rand_end = int(args.rand_fraction * RAND_MAX)
    edges = list(range(0, rand_end, args.chunk_size)) + [rand_end]
    ranges = [(edges[i], edges[i + 1] - 1) for i in range(len(edges) - 1)]
    print(f">> galaxy-context fetch rand_fraction={args.rand_fraction} "
          f"({args.dist_min_pc}-{args.dist_max_pc} pc, G<{args.gmax}): "
          f"{len(ranges)} chunks x {args.chunk_size:,}, {args.workers} workers",
          flush=True)

    server, url = probe(ranges[0][0], ranges[0][1], args)
    if url is None:
        sys.exit(">> no TAP server responded; try again later.")

    t0 = time.time()
    results, failed = fetch_ranges(url, ranges, args)

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

    cat = pd.concat(results.values(), ignore_index=True).drop_duplicates("source_id")
    cat.to_csv(args.out, index=False)
    got = len(ranges) - len(failed)
    with open(os.path.splitext(args.out)[0] + ".meta.json", "w") as fh:
        json.dump({"rand_fraction": args.rand_fraction,
                   "dist_min_pc": args.dist_min_pc,
                   "dist_max_pc": args.dist_max_pc,
                   "poe_min": args.poe_min, "gmax": args.gmax,
                   "n_rows": int(len(cat)),
                   "n_fetch_chunks": got,
                   "n_chunks_planned": len(ranges),
                   "n_chunks_missing": len(failed)}, fh, indent=2)
    print(f">> DONE: {len(cat):,} rows from {got}/{len(ranges)} chunks -> "
          f"{args.out} (+ .meta.json) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
