"""
pipeline.py
===========
Orchestrator. Ties the catalog, orbital Monte Carlo, and photometry together
and rolls the per-draw detections up into completeness and Drake-equation
yield estimates.

Flow
----
1. load + select HZ-relevant Sun-like stars               (catalog.py)
2. sample orbital geometry ONCE per star                  (montecarlo.py)
3. for each aperture x contrast scenario:                 (photometry.py)
       per-star detection probability p_det = <detect>_draws
       -> completeness, detectable-star counts, expected yield
4. connect to Drake terms via eta_Earth occurrence.

Key aggregates (per aperture x scenario)
-----------------------------------------
completeness        mean over stars of p_det   (prob an Earth analog in the HZ
                    is imaged, averaged over orbital geometry)
n_stars_accessible  number of stars with p_det > `accessible_threshold`
E_yield_sample      eta_Earth * sum(p_det)     (expected detected HZ analogs in
                    the simulated sample)
E_yield_local       E_yield_sample / rand_fraction   (extrapolated to the full
                    local population the DR3 slice represents)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

import catalog
import montecarlo
import photometry
from config import RunConfig


@dataclass
class PipelineResult:
    stars: pd.DataFrame          # selected stars + HZ geometry
    sweep_summary: pd.DataFrame  # one row per (aperture, scenario)
    per_star: pd.DataFrame       # long format: star x (aperture, scenario) p_det
    meta: dict


def run(cfg: RunConfig, base_dir: str = ".", test_mode: bool = False,
        accessible_threshold: float = 0.05, n_bootstrap: int = 0,
        ci: tuple = (16.0, 84.0), verbose: bool = True) -> PipelineResult:
    rng = np.random.default_rng(cfg.montecarlo.seed)

    # --- 1. catalog + selection ---------------------------------------
    raw = catalog.load_catalog(cfg.survey, base_dir=base_dir)
    stars, n_full = catalog.select_stars(raw, cfg.survey, cfg.montecarlo,
                                         test_mode=test_mode, rng=rng)
    n_stars = len(stars)                       # simulated (may be subsampled)
    if n_stars == 0:
        raise RuntimeError("No stars survived selection cuts.")
    if verbose:
        extra = f" (simulating {n_stars})" if n_stars != n_full else ""
        print(f">> {len(raw)} stars in shell -> {n_full} HZ-relevant selected"
              f"{extra}")

    # --- 2. orbital Monte Carlo (once) --------------------------------
    orb = montecarlo.sample_orbits(
        hz_inner_m=stars["hz_inner_m"].to_numpy(),
        hz_outer_m=stars["hz_outer_m"].to_numpy(),
        distance_m=stars["distance_m"].to_numpy(),
        mc=cfg.montecarlo, rng=rng,
    )
    if verbose:
        print(f">> sampled {orb.shape[1]} orbits x {orb.shape[0]} stars "
              f"= {orb.shape[0]*orb.shape[1]:,} draws")

    g_mag = stars["phot_g_mean_mag"].to_numpy()
    lum = stars["lum_flame"].to_numpy()
    frac = cfg.survey.rand_fraction or 1.0

    # observation mode -> effective bandwidth, SNR threshold, exposure
    bw_frac, target_snr, exposure_s, mode_label = cfg.observation.effective(cfg.instrument)
    if verbose:
        if exposure_s < 3600:
            tstr = f"{exposure_s / 60:.0f} min"
        elif exposure_s < 86400:
            tstr = f"{exposure_s / 3600:.1f} h"
        else:
            tstr = f"{exposure_s / 86400:.1f} d"
        print(f">> mode: {mode_label}, t_exp = {tstr}")

    # --- 3. sweep aperture x contrast ---------------------------------
    summary_rows = []
    per_star_rows = []
    pdet_store = {}                                     # (D, scenario) -> p_det array
    for D_m in cfg.apertures_m:
        for scen in cfg.contrast_scenarios:
            det = photometry.detectability(
                orb, g_mag, lum, D_m, scen.contrast_floor,
                exposure_s, cfg.instrument,
                bandwidth_frac=bw_frac, target_snr=target_snr,
            )
            p_det = det["detect"].mean(axis=1)          # (n_stars,)
            mean_snr = np.nanmean(det["snr"], axis=1)
            wa_frac = det["within_wa"].mean(axis=1)     # geometry-only completeness
            pdet_store[(D_m, scen.name)] = p_det

            completeness = float(p_det.mean())
            n_access = int((p_det > accessible_threshold).sum())
            # yield extrapolates completeness to the FULL selected population
            # (n_full), then to the local volume (/frac) -- correct under subsampling
            e_yield_sample = float(cfg.drake.eta_earth * completeness * n_full)

            per_star_rows.append(pd.DataFrame({
                "source_id": stars["source_id"].to_numpy(),
                "aperture_m": D_m,
                "scenario": scen.name,
                "p_det": p_det,
                "mean_snr": mean_snr,
                "wa_frac": wa_frac,
            }))

            summary_rows.append({
                "aperture_m": D_m,
                "scenario": scen.name,
                "mode": cfg.observation.mode,
                "exposure_s": exposure_s,
                "contrast_floor": scen.contrast_floor,
                "iwa_mas": det["iwa_rad"] * photometry.ARCSEC_PER_RAD * 1e3,
                "completeness": completeness,
                "wa_completeness": float(wa_frac.mean()),
                "n_stars_sim": n_stars,
                "n_selected_full": n_full,
                "n_accessible": n_access,
                "frac_accessible": n_access / n_stars,
                "eta_earth": cfg.drake.eta_earth,
                "E_yield_sample": e_yield_sample,
                "E_yield_local": e_yield_sample / frac,
                "n_stars_local_est": n_full / frac,
            })

    sweep = pd.DataFrame(summary_rows)
    per_star = pd.concat(per_star_rows, ignore_index=True)

    # --- 3b. bootstrap confidence intervals over the stellar sample ----
    if n_bootstrap > 0:
        sweep = _bootstrap_ci(sweep, pdet_store, n_stars, n_full, frac,
                              cfg.drake.eta_earth, accessible_threshold,
                              n_bootstrap, ci, seed=cfg.montecarlo.seed + 1,
                              verbose=verbose)

    meta = {
        "n_shell": int(len(raw)),
        "n_selected": n_stars,
        "n_bootstrap": n_bootstrap,
        "ci_percentiles": list(ci),
        "rand_fraction": frac,
        "n_draws": cfg.montecarlo.n_draws,
        "mode": cfg.observation.mode,
        "mode_label": mode_label,
        "exposure_s": exposure_s,
        "test_mode": test_mode,
        "accessible_threshold": accessible_threshold,
        "config": cfg.to_dict(),
    }
    if verbose:
        _print_headline(sweep, cfg)
    return PipelineResult(stars=stars, sweep_summary=sweep,
                          per_star=per_star, meta=meta)


def _bootstrap_ci(sweep: pd.DataFrame, pdet_store: dict, n_stars: int,
                  n_full: int, frac: float, eta_earth: float, thr: float,
                  n_boot: int, ci: tuple, seed: int,
                  verbose: bool = True) -> pd.DataFrame:
    """Resample the stellar sample WITH REPLACEMENT `n_boot` times and recompute
    the aggregates, giving confidence intervals on completeness / accessible /
    yield. The per-star p_det is fixed (Monte Carlo already done), so this only
    resamples star indices -- fast.

    Returns `sweep` with *_lo / *_hi / *_mean columns appended per metric.
    """
    if verbose:
        print(f">> bootstrap: {n_boot} resamples of {n_stars} stars "
              f"(CI = {ci[0]:g}-{ci[1]:g} percentile)")
    rng = np.random.default_rng(seed)
    lo_p, hi_p = ci
    add = {c: [] for c in (
        "completeness_mean", "completeness_lo", "completeness_hi",
        "frac_accessible_lo", "frac_accessible_hi",
        "E_yield_local_mean", "E_yield_local_lo", "E_yield_local_hi",
    )}
    for _, row in sweep.iterrows():
        pdet = pdet_store[(row["aperture_m"], row["scenario"])]
        # (n_boot, n_stars) index draw with replacement
        idx = rng.integers(0, n_stars, size=(n_boot, n_stars))
        res = pdet[idx]                                  # resampled p_det
        comp = res.mean(axis=1)
        frac_acc = (res > thr).mean(axis=1)
        yield_local = eta_earth * comp * n_full / frac
        add["completeness_mean"].append(float(comp.mean()))
        add["completeness_lo"].append(float(np.percentile(comp, lo_p)))
        add["completeness_hi"].append(float(np.percentile(comp, hi_p)))
        add["frac_accessible_lo"].append(float(np.percentile(frac_acc, lo_p)))
        add["frac_accessible_hi"].append(float(np.percentile(frac_acc, hi_p)))
        add["E_yield_local_mean"].append(float(yield_local.mean()))
        add["E_yield_local_lo"].append(float(np.percentile(yield_local, lo_p)))
        add["E_yield_local_hi"].append(float(np.percentile(yield_local, hi_p)))
    for c, vals in add.items():
        sweep[c] = vals
    return sweep


def save(result: PipelineResult, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    result.sweep_summary.to_csv(os.path.join(out_dir, "sweep_summary.csv"), index=False)
    result.per_star.to_csv(os.path.join(out_dir, "per_star_pdet.csv"), index=False)
    star_cols = ["source_id", "ra", "dec", "distance_pc", "teff_gspphot",
                 "logg_gspphot", "lum_flame", "phot_g_mean_mag",
                 "hz_inner_m", "hz_outer_m"]
    result.stars[star_cols].to_csv(os.path.join(out_dir, "selected_stars.csv"), index=False)
    with open(os.path.join(out_dir, "run_meta.json"), "w") as fh:
        json.dump(result.meta, fh, indent=2, default=str)


def _print_headline(sweep: pd.DataFrame, cfg: RunConfig) -> None:
    _, _, exp, label = cfg.observation.effective(cfg.instrument)
    print(f"\n=== Drake-term yield summary  [{label}]"
          f"  (eta_Earth={cfg.drake.eta_earth}, t_exp={exp:.0f}s) ===")
    cols = ["aperture_m", "scenario", "completeness", "n_accessible",
            "E_yield_sample", "E_yield_local"]
    with pd.option_context("display.float_format", lambda v: f"{v:,.4g}"):
        print(sweep[cols].to_string(index=False))
