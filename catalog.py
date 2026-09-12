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


def fetch_catalog(survey: SurveyConfig, rand_fraction: float, out_path: str) -> pd.DataFrame:
    """Query Gaia DR3 for a random `rand_fraction` slice within the distance
    shell and save it to `out_path` (+ a .meta.json). Requires internet."""
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
    job = Gaia.launch_job_async(query, dump_to_file=False)
    df = job.get_results().to_pandas()
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
