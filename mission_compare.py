#!/usr/bin/env python3
"""
mission_compare.py  --  validate our yield model against HWO / LUVOIR / HabEx.

Runs the SAME orbital-MC + photometry pipeline on a COMPLETE nearby-star census
(gaia_nearby.csv from fetch_nearby.py; rand_fraction = 1.0, so the yield is a
direct count -- no volume extrapolation) at the mission apertures, and overlays
the published concept-study exo-Earth yields.

This is a validation, not a fit: we use generic instrument parameters and check
our INDEPENDENT model lands in the right ballpark / reproduces the aperture
scaling. Agreement at large aperture (LUVOIR-A) is the key check.

    python3 mission_compare.py                       # 1e-10, 12 h, k_OWA=32
    python3 mission_compare.py --exp_min 1440 --contrast 1e-10
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RunConfig, InstrumentConfig, MonteCarloConfig, AU_M, R_EARTH_M
import catalog
import montecarlo
import photometry
from sensitivity import PARAMS, BASELINE, draw_params

ORBIT_SEED = 12345          # common random numbers across parameter draws

# concept-study values  (aperture [m], value, kind)
#   kind = "yield"       -> computed exo-Earth yield from the mission's yield code
#   kind = "requirement" -> mission design target, NOT a computed yield
LIT = {
    "HabEx":    (4.0,  8,  "yield"),        # HabEx final report (Gaudi+ 2020)
    "HWO":      (6.0,  25, "requirement"),  # Astro2020 decadal design target
    "LUVOIR-B": (8.0,  28, "yield"),        # LUVOIR final report (2019)
    "LUVOIR-A": (15.0, 54, "yield"),        # LUVOIR final report (2019)
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default="gaia_nearby.csv")
    p.add_argument("--contrast", type=float, default=1e-10)
    p.add_argument("--kowa", type=float, default=32.0)
    p.add_argument("--exp_min", type=float, default=720.0,
                   help="integration per target [min] (single-visit proxy)")
    p.add_argument("--n_draws", type=int, default=2000)
    p.add_argument("--mc", type=int, default=400,
                   help="parameter Monte Carlo draws for systematic error bars "
                        "(0 = baseline only)")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--replot", action="store_true",
                   help="skip the Monte Carlo and re-draw the figure from the "
                        "cached results (instant; for styling tweaks)")
    p.add_argument("--n_curve", type=int, default=24,
                   help="apertures for the smooth curve")
    p.add_argument("--dmin", type=float, default=3.0)
    p.add_argument("--dmax", type=float, default=20.0)
    p.add_argument("--out_dir", default="results/mission_compare")
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cache = os.path.join(out_dir, "mc_cache.npz")

    def _compute():
        cfg = RunConfig()
        cfg.survey.catalog_path = args.catalog
        cfg.survey.dist_min_pc = 0.0            # keep the whole nearby census
        cfg.survey.dist_max_pc = 1e6
        cfg.montecarlo.n_draws = args.n_draws
        rng = np.random.default_rng(cfg.montecarlo.seed)

        raw = catalog.load_catalog(cfg.survey, base_dir=base_dir)
        stars, n_full = catalog.select_stars(raw, cfg.survey, cfg.montecarlo,
                                              test_mode=False, rng=rng)
        frac = float(cfg.survey.rand_fraction or 1.0)
        mult = n_full / frac
        print(f">> {len(stars)} FGK dwarfs in the nearby census "
              f"(rand_fraction={frac}, eta_earth={cfg.drake.eta_earth})", flush=True)
        print(f">> distances: {stars['distance_pc'].min():.1f}-"
              f"{stars['distance_pc'].max():.1f} pc | integration {args.exp_min/60:g} h,"
              f" contrast {args.contrast:.0e}, k_OWA {args.kowa:g}", flush=True)

        sqrtL = np.sqrt(stars["lum_flame"].to_numpy())
        g = stars["phot_g_mean_mag"].to_numpy()
        lum = stars["lum_flame"].to_numpy()
        dist_m = stars["distance_m"].to_numpy()

        def yields_for(pp, apertures):
            mc = MonteCarloConfig(); mc.n_draws = args.n_draws; mc.seed = ORBIT_SEED
            mc.geometric_albedo = pp["geometric_albedo"]
            mc.planet_radius_m = R_EARTH_M * pp["radius_factor"]
            hz_in = pp["hz_inner_au"] * sqrtL * AU_M
            hz_out = pp["hz_outer_au"] * sqrtL * AU_M
            orb = montecarlo.sample_orbits(hz_in, hz_out, dist_m, mc,
                                           rng=np.random.default_rng(ORBIT_SEED))
            inst = InstrumentConfig()
            inst.eta_inst = pp["eta_inst"]; inst.eta_coron = pp["eta_coron"]
            inst.sigma_sys = pp["sigma_sys"]; inst.nzodi_level = pp["nzodi_level"]
            inst.k_owa = None if args.kowa <= 0 else args.kowa
            bw = inst.bandwidth
            out = np.empty(len(apertures))
            for k, D in enumerate(apertures):
                det = photometry.detectability(orb, g, lum, float(D), args.contrast,
                                               args.exp_min * 60.0, inst,
                                               bandwidth_frac=bw,
                                               target_snr=cfg.observation.snr_detect)
                out[k] = pp["eta_earth"] * det["detect"].mean() * mult
            return out

        aps = np.geomspace(args.dmin, args.dmax, args.n_curve)
        mission_D = np.array([D for _, (D, _, _) in LIT.items()])
        all_ap = np.concatenate([aps, mission_D])
        base = yields_for(BASELINE, all_ap)
        Y = None
        if args.mc > 0:
            print(f">> parameter MC: {args.mc} draws for systematic error bars ...",
                  flush=True)
            draws = draw_params(args.mc, args.seed)
            Y = np.empty((args.mc, len(all_ap)))
            t0 = time.time()
            for i in range(args.mc):
                Y[i] = yields_for({n: float(draws[n][i]) for n in PARAMS}, all_ap)
                if (i + 1) % max(1, args.mc // 20) == 0:
                    print(f"   {i+1}/{args.mc}  ({time.time()-t0:.0f}s)", flush=True)
        return aps, mission_D, base, Y

    # ---- compute, or load cached results for an instant re-plot ----
    if args.replot and os.path.exists(cache):
        z = np.load(cache)
        aps, mission_D, base = z["aps"], z["mission_D"], z["base"]
        Y = z["Y"] if "Y" in z.files else None
        print(f">> replot from cache: {cache}", flush=True)
    else:
        aps, mission_D, base, Y = _compute()
        save = {"aps": aps, "mission_D": mission_D, "base": base}
        if Y is not None:
            save["Y"] = Y
        np.savez(cache, **save)

    na = len(aps)
    curve, base_miss = base[:na], base[na:]
    lo = hi = mlo = mhi = None
    if Y is not None:
        med = np.median(Y, axis=0)
        lo = np.percentile(Y, 16, axis=0)[:na]
        hi = np.percentile(Y, 84, axis=0)[:na]
        curve, base_miss = med[:na], med[na:]
        mlo = np.percentile(Y[:, na:], 16, axis=0)
        mhi = np.percentile(Y[:, na:], 84, axis=0)

    print("\n   mission     D[m]   our yield   literature")
    rows = []
    for j, (name, (D, ylit, kind)) in enumerate(LIT.items()):
        yo = base_miss[j]
        band = f"  [{mlo[j]:.1f}, {mhi[j]:.1f}]" if mlo is not None else ""
        rows.append((name, D, yo, ylit, kind,
                     None if mlo is None else (mlo[j], mhi[j])))
        print(f"   {name:<10} {D:>5.0f}   {yo:>9.1f}   {ylit:>10}{band}  ({kind})")

    # ---- figure (single-column, fonts/aspect matched to the other figures) ----
    plt.rcParams.update({"axes.titlesize": 10, "axes.labelsize": 11,
                         "xtick.labelsize": 9, "ytick.labelsize": 9})
    fig, ax = plt.subplots(figsize=(3.7, 3.27))
    if lo is not None:
        ax.fill_between(aps, lo, hi, color="#08306b", alpha=0.18, linewidth=0)
    ax.plot(aps, curve, "-", color="#08306b", lw=1.8, label="this work")
    for name, D, yo, ylit, kind, mb in rows:
        if mb is not None:
            ax.errorbar([D], [yo], yerr=[[yo-mb[0]], [mb[1]-yo]], fmt="o", ms=4,
                        color="#08306b", ecolor="#08306b", elinewidth=1.1,
                        capsize=2.5, zorder=5)
        else:
            ax.scatter([D], [yo], s=30, color="#08306b", zorder=5)
        if kind == "requirement":            # same star, hollow inside (label says req.)
            ax.scatter([D], [ylit], s=75, marker="*", facecolors="white",
                       edgecolors="black", linewidths=0.6, zorder=6)
            ax.annotate(f"{name} (req.)", xy=(D, ylit), xytext=(-4, -12),
                        textcoords="offset points", ha="right", fontsize=6.5)
        else:
            ax.scatter([D], [ylit], s=75, marker="*", color="#F5821F",
                       edgecolors="black", linewidths=0.4, zorder=6)
            ax.annotate(name, xy=(D, ylit), xytext=(3, 5),
                        textcoords="offset points", fontsize=6.5)
    ax.scatter([], [], s=30, color="#08306b", label="this work (mission $D$)")
    ax.scatter([], [], s=75, marker="*", color="#F5821F", edgecolors="black",
               linewidths=0.4, label="concept study")
    ax.set_xscale("log"); ax.set_yscale("log")
    import matplotlib.ticker as mticker
    ax.set_xticks([3, 4, 5, 6, 8, 10, 15, 20])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("aperture diameter D [m]")
    ax.set_ylabel("expected HZ Earth yield")
    ax.set_title("Comparison with concept studies")
    ax.grid(True, which="major", alpha=0.2)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    out = os.path.join(out_dir, "mission_compare.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\n>> wrote {out}  (error bars = 1sigma parameter systematic)", flush=True)


if __name__ == "__main__":
    main()
