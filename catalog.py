"""
catalog.py
==========
Build the stellar sample from Gaia DR3.

Two ways to get stars:
  * load_catalog()   -- read a cached CSV pull (fast, offline; used for tests)
  * fetch_catalog()  -- query the Gaia archive directly (needs astroquery + net)

Then select_stars() applies the Sun-like / HZ-relevant cuts and attaches the
habitable-zone geometry each star needs for the Monte Carlo.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from config import SurveyConfig, MonteCarloConfig, AU_M, PC_M

REQUIRED_COLUMNS = [
    "source_id", "ra", "dec", "parallax", "phot_g_mean_mag",
    "teff_gspphot", "logg_gspphot", "lum_flame",
]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_catalog(survey: SurveyConfig, base_dir: str = ".") -> pd.DataFrame:
    """Load a cached DR3 CSV and clean rows with missing core parameters.

    If a `<catalog>.meta.json` sits next to the CSV, its `rand_fraction` is
    written back onto `survey` so volume extrapolation stays correct.
    """
    path = _resolve(survey.catalog_path, base_dir)
    df = pd.read_csv(path)

    # Pick up the true sampled fraction from the meta file if present.
    meta_path = _meta_path(path)
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
        if "rand_fraction" in meta:
            survey.rand_fraction = float(meta["rand_fraction"])

    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    df = df[(df["parallax"] > 0) & (df["teff_gspphot"] > 0) & (df["lum_flame"] > 0)].copy()
    df["distance_pc"] = 1000.0 / df["parallax"]
    df = df[df["distance_pc"].between(survey.dist_min_pc, survey.dist_max_pc)].copy()
    return df.reset_index(drop=True)


# TAP mirrors that host the SAME gaiadr3 schema (gaia_source +
# astrophysical_parameters), so our query runs against them unchanged. Tried in
# order when the ESA archive is unreachable.
DEFAULT_MIRRORS = ["https://gaia.ari.uni-heidelberg.de/tap"]  # ARI-Heidelberg


def _is_transient(exc) -> bool:
    """True for errors worth retrying (server 5xx, connection/timeout);
    False for clear client-side errors (4xx / bad query / missing table) that
    won't self-heal -- those make us give up on this server and try the next."""
    s = str(exc).lower()
    for kw in ("not found", "unknown table", "does not exist", "undefined",
               "syntax", "unrecognized"):
        if kw in s:
            return False
    for code in ("400", "401", "403", "404", "405", "413"):
        if code in str(exc):
            return False
    return True


def _run_query(client, query, name, attempts, base_wait, max_wait):
    """Run `query` on one TAP client with exponential-backoff retries. Returns a
    DataFrame, or raises after `attempts` transient failures / on a hard error."""
    for attempt in range(1, attempts + 1):
        try:
            job = client.launch_job_async(query, dump_to_file=False)
            return job.get_results().to_pandas()
        except Exception as exc:                       # noqa: BLE001 (want broad here)
            if not _is_transient(exc):
                print(f"   {name}: hard error ({type(exc).__name__}: {exc}); "
                      f"skipping this server", flush=True)
                raise
            if attempt >= attempts:
                print(f"   {name}: still failing after {attempts} attempts",
                      flush=True)
                raise
            wait = min(base_wait * 2 ** (attempt - 1), max_wait)
            print(f"   {name}: attempt {attempt}/{attempts} failed "
                  f"({type(exc).__name__}: {exc}); retrying in {wait:.0f}s ...",
                  flush=True)
            time.sleep(wait)


def fetch_catalog(survey: SurveyConfig, rand_fraction: float, out_path: str,
                  max_attempts: int = 100, primary_attempts: int = 5,
                  base_wait: float = 15.0, max_wait: float = 300.0,
                  mirrors=None) -> pd.DataFrame:
    """Query Gaia DR3 for a random `rand_fraction` slice within the distance
    shell and save it to `out_path` (+ a .meta.json). Requires internet.

    Robust to Gaia archive outages:
      * transient errors (5xx / connection / timeout) are retried with
        exponential backoff (capped at `max_wait`);
      * if the ESA archive keeps failing (after `primary_attempts`), it falls
        back to TAP mirror(s) with the same gaiadr3 schema and retries there
        (up to `max_attempts` each).
    Pass mirrors=[] to disable the fallback (ESA only); mirrors=None uses
    DEFAULT_MIRRORS. Only genuine client/schema errors abort a given server.
    """
    from astroquery.gaia import Gaia  # imported lazily so offline tests don't need it

    parallax_max = 1000.0 / survey.dist_min_pc
    parallax_min = 1000.0 / survey.dist_max_pc
    rand_max = 1_800_000_000
    rand_end = int(rand_fraction * rand_max)

    query = f"""
    SELECT g.source_id, g.ra, g.dec, g.parallax, g.parallax_over_error,
           g.phot_g_mean_mag, g.random_index,
           ap.teff_gspphot, ap.logg_gspphot, ap.lum_flame,
           ap.mass_flame, ap.radius_flame
    FROM gaiadr3.gaia_source AS g
    JOIN gaiadr3.astrophysical_parameters AS ap ON g.source_id = ap.source_id
    WHERE g.parallax BETWEEN {parallax_min} AND {parallax_max}
      AND g.parallax_over_error > 5
      AND ap.lum_flame IS NOT NULL
      AND ap.teff_gspphot IS NOT NULL
      AND ap.logg_gspphot IS NOT NULL
      AND g.random_index BETWEEN 0 AND {rand_end}
    """

    mirror_list = DEFAULT_MIRRORS if mirrors is None else list(mirrors)
    # (name, client, attempts). ESA gets the full budget when there is no mirror
    # to fall back to, else just a few tries before switching.
    esa_attempts = primary_attempts if mirror_list else max_attempts
    backends = [("ESA archive", Gaia, esa_attempts)]
    if mirror_list:
        from astroquery.utils.tap.core import TapPlus
        for url in mirror_list:
            backends.append((f"mirror {url}", TapPlus(url=url), max_attempts))

    df = None
    last_exc = None
    for name, client, attempts in backends:
        print(f">> querying {name} ({attempts} attempts max) ...", flush=True)
        try:
            df = _run_query(client, query, name, attempts, base_wait, max_wait)
            print(f">> got {len(df)} rows from {name}", flush=True)
            break
        except Exception as exc:                       # noqa: BLE001
            last_exc = exc
            continue
    if df is None:
        raise RuntimeError(f"all TAP servers failed; last error: {last_exc}")

    df.to_csv(out_path, index=False)
    with open(_meta_path(out_path), "w") as fh:
        json.dump({"rand_fraction": rand_fraction,
                   "dist_min_pc": survey.dist_min_pc,
                   "dist_max_pc": survey.dist_max_pc,
                   "n_rows": len(df)}, fh, indent=2)
    survey.rand_fraction = rand_fraction
    return df


# --------------------------------------------------------------------------
# Selection + habitable-zone geometry
# --------------------------------------------------------------------------
def select_stars(df: pd.DataFrame, survey: SurveyConfig, mc: MonteCarloConfig,
                 test_mode: bool = False, rng: np.random.Generator | None = None):
    """Apply Sun-like main-sequence cuts and attach HZ geometry.

    Returns (sel, n_full):
      sel     -- DataFrame with distance_m, hz_inner_m, hz_outer_m per star
                 (subsampled to survey.test_max_stars in test_mode);
      n_full  -- number of stars passing the cuts BEFORE any subsampling.
    n_full is what absolute yields must extrapolate from: completeness is measured
    on the (representative) subsample, but the population is n_full stars.
    """
    sel = df[
        (df["logg_gspphot"] > survey.logg_min)
        & (df["teff_gspphot"].between(survey.teff_min_k, survey.teff_max_k))
        & (df["lum_flame"].between(survey.lum_min_lsun, survey.lum_max_lsun))
    ].copy()
    n_full = len(sel)

    if test_mode and survey.test_max_stars and len(sel) > survey.test_max_stars:
        rng = rng or np.random.default_rng(mc.seed)
        idx = rng.choice(len(sel), size=survey.test_max_stars, replace=False)
        sel = sel.iloc[np.sort(idx)].copy()

    sel = sel.reset_index(drop=True)
    sel["distance_m"] = sel["distance_pc"] * PC_M

    # HZ edges scale as sqrt(L/L_sun); expressed as orbital radii in metres.
    sqrtL = np.sqrt(sel["lum_flame"].to_numpy())
    sel["hz_inner_m"] = mc.hz_inner_au * sqrtL * AU_M
    sel["hz_outer_m"] = mc.hz_outer_au * sqrtL * AU_M
    return sel, n_full


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _resolve(path: str, base_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base_dir, path))


def _meta_path(csv_path: str) -> str:
    root, _ = os.path.splitext(csv_path)
    return root + ".meta.json"
