#!/usr/bin/env python3
"""
sensitivity.py
==============
SYSTEMATIC (model-parameter) uncertainty on the direct-imaging yield -- the
second error tier, complementing the bootstrap sampling error.

The bootstrap asked "did we get a lucky star sample?" (answer: negligible).
This asks "did we pick the right values for the model assumptions?" -- eta_Earth,
albedo, exozodi, speckle stability, HZ edges, throughput, planet radius -- each
of which has real literature uncertainty.

Two analyses, both at ONE reference design point (aperture, contrast, k_OWA,
exposure), which is where a headline yield is quoted:

  1. One-at-a-time (OAT): vary each parameter alone across its range, recompute
     the yield, record the swing -> a tornado ranking of "which knob matters".
  2. Global Monte Carlo: draw ALL parameters simultaneously from flat priors over
     their ranges, recompute the yield N times -> the combined systematic band.

Variance reduction: the orbital Monte Carlo uses COMMON RANDOM NUMBERS (a fixed
seed) across every evaluation, so the spread in the yield is driven by the
PARAMETERS, not by orbital sampling noise. eta_Earth and the population factor are
exact post-completeness multipliers, applied analytically.

Outputs (into --out_dir):
  oat.csv          per-parameter low/high yields and % swing
  mc_samples.csv   every MC draw: all parameters + completeness + yield
  summary.json     baseline yield, MC percentiles, per-source ranking
  tornado.png, mc_hist.png   (unless --no_plots)

    python3 sensitivity.py --aperture 15 --contrast 1e-10 --kowa 32 \
        --exp_min 360 --n_mc 1000
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from config import (RunConfig, InstrumentConfig, MonteCarloConfig,
                    AU_M, R_EARTH_M)
import catalog
import montecarlo
import photometry

MIN = 60.0
ORBIT_SEED = 12345          # fixed -> common random numbers across evaluations

# parameter: (baseline, low, high).  Flat priors over [low, high].
PARAMS = {
    "eta_earth":        (0.24, 0.10, 0.50),   # HZ Earth occurrence (Drake term)
    "geometric_albedo": (0.30, 0.20, 0.40),   # planet reflectivity
    "nzodi_level":      (3.0,  1.0,  10.0),    # exozodiacal dust [zodis]
    "sigma_sys":        (0.10, 0.05, 0.20),    # residual speckle stability
    "eta_inst":         (0.20, 0.15, 0.25),    # instrument throughput
    "eta_coron":        (0.30, 0.20, 0.40),    # coronagraph throughput
    "hz_inner_au":      (0.95, 0.84, 0.99),    # inner HZ edge (conservative<->optimistic)
    "hz_outer_au":      (1.67, 1.40, 1.77),    # outer HZ edge
    "radius_factor":    (1.0,  0.8,  1.2),     # planet radius in R_Earth
}
BASELINE = {k: v[0] for k, v in PARAMS.items()}


def _bar(done, total, t0, width=28):
    frac = done / max(total, 1)
    fill = int(width * frac)
    el = time.time() - t0
    eta = el / max(done, 1) * (total - done)
    return (f"[{done:>5d}/{total} {100*frac:5.1f}%] "
            f"|{'#'*fill}{'-'*(width-fill)}| "
            f"elapsed {el:5.0f}s  eta {eta:5.0f}s")


class Model:
    """Holds the fixed star sample and evaluates the yield for a parameter set."""
    def __init__(self, cfg, base_dir, max_stars, n_draws, design, mode):
        rng = np.random.default_rng(ORBIT_SEED)
        raw = catalog.load_catalog(cfg.survey, base_dir=base_dir)
        if max_stars:
            cfg.survey.test_max_stars = max_stars
        stars, n_full = catalog.select_stars(raw, cfg.survey, cfg.montecarlo,
                                              test_mode=bool(max_stars), rng=rng)
        self.n_full = int(n_full)
        self.frac = float(cfg.survey.rand_fraction or 1.0)
        self.sqrtL = np.sqrt(stars["lum_flame"].to_numpy())
        self.g = stars["phot_g_mean_mag"].to_numpy()
        self.lum = stars["lum_flame"].to_numpy()
        self.dist_m = stars["distance_m"].to_numpy()
        self.n_draws = n_draws
        self.design = design                    # dict D, contrast, kowa, exp_s
        self.base_inst = cfg.instrument
        self.mode = mode
        self.mult = self.n_full / self.frac     # population extrapolation factor
        print(f">> model: {len(stars):,} sim stars (of {self.n_full:,} full), "
              f"{n_draws} draws | design D={design['D']}m contrast={design['contrast']:.0e} "
              f"k_OWA={design['kowa']:g} t={design['exp_s']/60:g}min", flush=True)

    def completeness(self, p):
        """Fraction of viable candidates yielding a detectable HZ Earth analog
        (everything EXCEPT the eta_Earth multiplier), for parameter set p."""
        mc = MonteCarloConfig()
        mc.n_draws = self.n_draws
        mc.seed = ORBIT_SEED
        mc.geometric_albedo = p["geometric_albedo"]
        mc.planet_radius_m = R_EARTH_M * p["radius_factor"]
        hz_in = p["hz_inner_au"] * self.sqrtL * AU_M
        hz_out = p["hz_outer_au"] * self.sqrtL * AU_M
        rng = np.random.default_rng(ORBIT_SEED)
        orb = montecarlo.sample_orbits(hz_in, hz_out, self.dist_m, mc, rng=rng)

        inst = InstrumentConfig()
        inst.eta_inst = p["eta_inst"]
        inst.eta_coron = p["eta_coron"]
        inst.sigma_sys = p["sigma_sys"]
        inst.nzodi_level = p["nzodi_level"]
        inst.k_owa = None if self.design["kowa"] <= 0 else self.design["kowa"]

        bw = (1.0 / self.mode["R"]) if self.mode["name"] == "characterization" \
            else inst.bandwidth
        det = photometry.detectability(
            orb, self.g, self.lum, self.design["D"], self.design["contrast"],
            self.design["exp_s"], inst, bandwidth_frac=bw,
            target_snr=self.mode["snr"])
        return float(det["detect"].mean())

    def yield_of(self, p, comp=None):
        if comp is None:
            comp = self.completeness(p)
        return p["eta_earth"] * comp * self.mult, comp


def run_oat(model, out_dir):
    print(">> OAT sensitivity (vary each parameter alone) ...", flush=True)
    y0, c0 = model.yield_of(BASELINE)
    rows = []
    for i, (name, (base, lo, hi)) in enumerate(PARAMS.items(), 1):
        ylo, _ = model.yield_of({**BASELINE, name: lo})
        yhi, _ = model.yield_of({**BASELINE, name: hi})
        rows.append(dict(parameter=name, baseline=base, low=lo, high=hi,
                         Y_low=ylo, Y_high=yhi, Y0=y0,
                         swing_low_pct=100*(ylo-y0)/y0,
                         swing_high_pct=100*(yhi-y0)/y0,
                         abs_range=abs(yhi-ylo)))
        print(f"   [{i}/{len(PARAMS)}] {name:<17s} "
              f"Y: {ylo:8.2f} .. {yhi:8.2f}  (Y0={y0:.2f})", flush=True)
    df = pd.DataFrame(rows).sort_values("abs_range", ascending=False)
    df.to_csv(os.path.join(out_dir, "oat.csv"), index=False)
    return y0, c0, df


def run_mc(model, n_mc, seed, out_dir):
    print(f">> global Monte Carlo: {n_mc} parameter draws ...", flush=True)
    rng = np.random.default_rng(seed)
    names = list(PARAMS.keys())
    draws = {n: rng.uniform(PARAMS[n][1], PARAMS[n][2], size=n_mc) for n in names}
    ys = np.empty(n_mc)
    comps = np.empty(n_mc)
    t0 = time.time()
    step = max(1, n_mc // 50)
    for i in range(n_mc):
        p = {n: draws[n][i] for n in names}
        y, c = model.yield_of(p)
        ys[i] = y
        comps[i] = c
        if (i + 1) % step == 0 or i + 1 == n_mc:
            print("   " + _bar(i + 1, n_mc, t0), flush=True)
    out = pd.DataFrame(draws)
    out["completeness"] = comps
    out["yield"] = ys
    out.to_csv(os.path.join(out_dir, "mc_samples.csv"), index=False)
    return ys


def make_plots(y0, oat_df, ys, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # tornado
    d = oat_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    yv = np.arange(len(d))
    for k, (_, r) in enumerate(d.iterrows()):
        ax.barh(k, r["Y_high"] - r["Y_low"], left=min(r["Y_low"], r["Y_high"]),
                color="#F5821F", alpha=0.85)
    ax.axvline(y0, color="black", ls="--", lw=1, label=f"baseline = {y0:.1f}")
    ax.set_yticks(yv); ax.set_yticklabels(d["parameter"])
    ax.set_xlabel("expected yield"); ax.legend()
    ax.set_title("OAT sensitivity (tornado)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "tornado.png"), dpi=180)
    plt.close(fig)
    # MC histogram
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(ys, bins=40, color="#1f77b4", alpha=0.8)
    for q, ls in [(16, ":"), (50, "-"), (84, ":")]:
        ax.axvline(np.percentile(ys, q), color="black", ls=ls, lw=1.2)
    ax.set_xlabel("expected yield"); ax.set_ylabel("MC draws")
    ax.set_title("global parameter Monte Carlo")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "mc_hist.png"), dpi=180)
    plt.close(fig)
    print(">> wrote tornado.png, mc_hist.png", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aperture", type=float, default=15.0, help="reference D [m]")
    p.add_argument("--contrast", type=float, default=1e-10)
    p.add_argument("--kowa", type=float, default=32.0)
    p.add_argument("--exp_min", type=float, default=360.0)
    p.add_argument("--mode", choices=["detection", "characterization"],
                   default="detection")
    p.add_argument("--spectral_R", type=float, default=70.0)
    p.add_argument("--snr", type=float, default=10.0)
    p.add_argument("--n_mc", type=int, default=1000)
    p.add_argument("--max_stars", type=int, default=40000,
                   help="star subsample for speed (sampling noise is negligible)")
    p.add_argument("--n_draws", type=int, default=800)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--out_dir", default="results/sensitivity")
    p.add_argument("--no_plots", action="store_true")
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cfg = RunConfig()
    design = dict(D=args.aperture, contrast=args.contrast, kowa=args.kowa,
                  exp_s=args.exp_min * MIN)
    mode = dict(name=args.mode, R=args.spectral_R, snr=args.snr)
    model = Model(cfg, base_dir, args.max_stars, args.n_draws, design, mode)

    t0 = time.time()
    y0, c0, oat_df = run_oat(model, out_dir)
    ys = run_mc(model, args.n_mc, args.seed, out_dir)

    pct = {q: float(np.percentile(ys, q)) for q in (2.5, 16, 50, 84, 97.5)}
    summary = dict(
        design=design, mode=args.mode, n_mc=args.n_mc,
        n_sim=int(len(model.g)), n_full=model.n_full,
        baseline_yield=y0, baseline_completeness=c0,
        mc_mean=float(ys.mean()), mc_std=float(ys.std()),
        mc_median=pct[50], mc_p16=pct[16], mc_p84=pct[84],
        mc_p2p5=pct[2.5], mc_p97p5=pct[97.5],
        mc_rel_sigma=float(ys.std() / ys.mean()),
        oat_ranking=oat_df[["parameter", "abs_range"]].to_dict("records"),
    )
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    if not args.no_plots:
        try:
            make_plots(y0, oat_df, ys, out_dir)
        except Exception as e:
            print(f"   (plots skipped: {e})", flush=True)

    print(f"\n>> DONE in {time.time()-t0:.0f}s")
    print(f">> baseline yield Y0 = {y0:.2f}")
    print(f">> systematic band (68%): {pct[16]:.2f} .. {pct[84]:.2f}  "
          f"(median {pct[50]:.2f}, +/-{100*summary['mc_rel_sigma']:.0f}% 1-sigma)")
    print(f">> dominant parameter: {oat_df.iloc[0]['parameter']}")


if __name__ == "__main__":
    main()
