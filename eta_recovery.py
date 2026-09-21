#!/usr/bin/env python3
"""
eta_recovery.py
===============
Invert the yield to *measure* eta_Earth (the reviewer's request): instead of
asking "how much does the predicted yield move when eta_Earth varies", ask the
inverse -- "if the survey detects N_det planets, how precisely can eta_Earth be
recovered?".

For each aperture we know the effective number of searchable stars
    Sum_C = <completeness> * N_pop           (N_pop = n_full / rand_fraction)
so the expected detections at the true rate are
    N_det = eta_true * Sum_C .
A synthetic survey draws
    N_det ~ Poisson(eta_true * Sum_C)
and the analyst recovers
    eta_hat = N_det / Sum_C_model ,
where Sum_C_model carries the nuisance-parameter (albedo/radius/exozodi) modelling
uncertainty -- the CV(completeness) "floor" measured by sensitivity.py. Repeating
the survey many times gives the distribution of eta_hat, whose width is

    sigma_eta / eta  =  sqrt( 1/N_det              (statistical, Poisson)
                              + CV(Sum_C)^2 ) .    (systematic, nuisance)

Reads completeness from a cube (run_cube.py) and the nuisance floor from the
sensitivity summary.json. No re-simulation.

    python3 eta_recovery.py
    python3 eta_recovery.py --nuisance fixed      # statistical (Poisson) only
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cube", default="cube_fine.h5")
    p.add_argument("--summary", default="results/sensitivity/summary.json")
    p.add_argument("--contrast", type=float, default=1e-10)
    p.add_argument("--kowa", type=float, default=32.0)
    p.add_argument("--time_min", type=float, default=360.0)
    p.add_argument("--eta_true", type=float, default=0.24)
    p.add_argument("--n_trials", type=int, default=20000)
    p.add_argument("--nuisance", choices=["marginalize", "fixed"],
                   default="marginalize",
                   help="'marginalize' folds the nuisance (albedo/radius/exozodi) "
                        "floor into Sum_C; 'fixed' assumes it perfectly known "
                        "(statistical/Poisson limit only)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out_dir", default="results/eta_recovery")
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ---- completeness (-> Sum_C, N_det) from the cube ----
    with h5py.File(os.path.join(base_dir, args.cube), "r") as f:
        ap = f["axes/apertures_m"][:]
        con = f["axes/contrasts"][:]
        tmin = f["axes/times_min"][:]
        kowa = f["axes/kowa"][:]
        comp = f["completeness"][:]
        n_full = int(f.attrs["n_full"])
        frac = float(f.attrs["rand_fraction"])
    ci = int(np.argmin(np.abs(con - args.contrast)))
    ti = int(np.argmin(np.abs(tmin - args.time_min)))
    ki = int(np.argmin(np.abs(kowa - args.kowa)))
    mult = n_full / frac                                  # full local population

    # ---- nuisance floor CV(completeness) per aperture from sensitivity ----
    with open(os.path.join(base_dir, args.summary)) as fh:
        summ = json.load(fh)
    ap_floor = np.array([float(k) for k in summ["per_aperture"]])
    cv_floor = np.array([summ["per_aperture"][k]["rel_sigma_eta_removed"]
                         for k in summ["per_aperture"]])
    order = np.argsort(ap_floor)
    ap_floor, cv_floor = ap_floor[order], cv_floor[order]

    rng = np.random.default_rng(args.seed)
    print(f">> eta recovery: contrast {args.contrast:.0e}, k_OWA {args.kowa:g}, "
          f"t={args.time_min:g} min, eta_true={args.eta_true}, "
          f"nuisance={args.nuisance}\n", flush=True)
    print(f"{'D[m]':>6} {'Ndet':>10} {'stat%':>7} {'syst%':>7} {'total%':>7} "
          f"{'eta_hat 16-84':>18}")

    Ds, Ndet_a, stat_a, syst_a, tot_a = [], [], [], [], []
    lo_a, med_a, hi_a = [], [], []
    for D, cv in zip(ap_floor, cv_floor):
        ai = int(np.argmin(np.abs(ap - D)))
        Cbar = comp[ai, ci, ti, ki]
        SumC = Cbar * mult
        Ndet = args.eta_true * SumC
        if Ndet <= 0:
            continue
        # repeated synthetic surveys
        nd = rng.poisson(Ndet, args.n_trials).astype(float)
        if args.nuisance == "marginalize" and cv > 0:
            # log-normal model error on Sum_C, median-unbiased
            SumC_model = SumC * np.exp(cv * rng.standard_normal(args.n_trials))
        else:
            SumC_model = SumC
        eta_hat = nd / SumC_model
        med = np.median(eta_hat)
        lo = np.percentile(eta_hat, 16); hi = np.percentile(eta_hat, 84)
        stat = 1.0 / np.sqrt(Ndet)
        syst = cv if args.nuisance == "marginalize" else 0.0
        tot = eta_hat.std() / max(med, 1e-9)
        Ds.append(D); Ndet_a.append(Ndet)
        stat_a.append(stat); syst_a.append(syst); tot_a.append(tot)
        lo_a.append(lo); med_a.append(med); hi_a.append(hi)
        print(f"{D:>6.0f} {Ndet:>10.0f} {100*stat:>7.2f} {100*syst:>7.1f} "
              f"{100*tot:>7.1f}   [{lo:>6.3f}, {hi:>6.3f}]")

    Ds = np.array(Ds); Ndet_a = np.array(Ndet_a)
    stat_a = np.array(stat_a); syst_a = np.array(syst_a); tot_a = np.array(tot_a)
    lo_a = np.array(lo_a); med_a = np.array(med_a); hi_a = np.array(hi_a)

    plt.rcParams.update({"axes.titlesize": 10, "axes.labelsize": 11,
                         "xtick.labelsize": 9, "ytick.labelsize": 9})

    # ---- FIGURE 1: recovered eta_hat with 16-84 band ----
    fig, ax = plt.subplots(figsize=(3.7, 3.27))
    ax.fill_between(Ds, lo_a, hi_a, color="#F5821F", alpha=0.30, label="16-84%")
    ax.plot(Ds, med_a, "o-", color="#9a4a00", lw=1.8, ms=3.5, label="median $\\hat\\eta_\\oplus$")
    ax.axhline(args.eta_true, color="black", ls="--", lw=1,
               label=f"true $\\eta_\\oplus$={args.eta_true:g}")
    ax.set_xlabel("aperture D [m]"); ax.set_ylabel("recovered $\\eta_\\oplus$")
    ax.set_title("Recovered $\\eta_\\oplus$ vs aperture")
    ax.grid(True, which="major", alpha=0.2); ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "eta_recovery.png"), dpi=200)
    plt.close(fig)

    # ---- FIGURE 2: precision decomposition ----
    fig, ax = plt.subplots(figsize=(3.7, 3.27))
    ax.plot(Ds, 100*stat_a, "o:", color="#1f77b4", lw=1.6, ms=4, alpha=0.7,
            label="statistical $1/\\sqrt{N_\\mathrm{det}}$")
    if args.nuisance == "marginalize":
        ax.plot(Ds, 100*syst_a, "o--", color="#2ca02c", lw=1.6, ms=4, alpha=0.7,
                label="systematic (nuisance)")
    ax.plot(Ds, 100*tot_a, "o-", color="#d62728", lw=2, ms=4, alpha=0.62,
            label="total (recovered)")
    # red value boxes on the total curve
    off = {10: (24, -3), 100: (0, 12), 250: (0, 12),
           500: (0, 13), 750: (0, 13), 1000: (-12, 13)}
    for D, tv in zip(Ds, 100*tot_a):
        o = off.get(int(round(D)))
        if o is not None:
            ax.annotate(f"{tv:.1f}%", xy=(D, tv), xytext=o,
                        textcoords="offset points", ha="center", fontsize=7,
                        color="#a01010",
                        arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.9),
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec="#d62728", lw=0.9, alpha=0.95))
    ax.set_xlabel("aperture D [m]")
    ax.set_ylabel("$\\sigma_{\\eta_\\oplus}/\\eta_\\oplus$  [%]")
    ax.set_title("Fractional uncertainty in recovered\n"
                 "$\\eta_\\oplus$ versus aperture", fontsize=9.5)
    ax.set_ylim(0, max(100*tot_a) * 1.16)
    ax.grid(True, which="major", alpha=0.2); ax.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "eta_precision.png"), dpi=200)
    plt.close(fig)

    print(f"\n>> wrote eta_recovery.png, eta_precision.png -> {out_dir}")


if __name__ == "__main__":
    main()
