#!/usr/bin/env python3
"""
sensitivity.py
==============
SYSTEMATIC (model-parameter) uncertainty on the direct-imaging yield -- the
second error tier, complementing the bootstrap sampling error. Now swept over
SEVERAL aperture diameters in one pass.

The bootstrap asked "did we get a lucky star sample?" (negligible). This asks
"did we pick the right values for the model assumptions?" -- eta_Earth, albedo,
exozodi, speckle stability, HZ edges, throughput, planet radius.

Two analyses, at EACH aperture in --apertures:
  1. One-at-a-time (OAT): vary each parameter alone across its range, recompute
     the yield -> a tornado ranking of "which knob matters".
  2. Global Monte Carlo: draw ALL parameters simultaneously from flat priors ->
     the combined systematic band.

Efficiency: orbital geometry is aperture-independent, so each parameter draw
samples orbits ONCE and evaluates the photometry at every aperture. Common
random numbers (fixed orbital seed) make the spread parameter-driven, not
orbital-MC noise. eta_Earth is an exact post-completeness multiplier.

Run (single node):        python3 sensitivity.py --apertures 10 20 50 100 250 500 750 1000
Parallel (CARC array):    --shard i --nshards N --tmp parts_sens ; then --combine
Re-plot from saved CSVs:  python3 sensitivity.py --plots_only --out_dir results/sensitivity
"""

from __future__ import annotations

import argparse
import glob
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
ORBIT_SEED = 12345

PARAMS = {
    "eta_earth":        (0.24, 0.10, 0.50),
    "geometric_albedo": (0.20, 0.10, 0.35),   # Earth-twin V-band A_g (central 0.20)
    "nzodi_level":      (3.0,  1.0,  10.0),
    "sigma_sys":        (0.10, 0.05, 0.20),
    "eta_inst":         (0.20, 0.15, 0.25),
    "eta_coron":        (0.30, 0.20, 0.40),
    "hz_inner_au":      (0.95, 0.84, 0.99),
    "hz_outer_au":      (1.67, 1.40, 1.77),
    "radius_factor":    (1.0,  0.8,  1.2),
}
BASELINE = {k: v[0] for k, v in PARAMS.items()}

# symbolic labels for figures
LABELS = {
    "eta_earth":        r"$\eta_\oplus$",
    "geometric_albedo": r"$A_g$",
    "nzodi_level":      r"$n_\mathrm{zodi}$",
    "sigma_sys":        r"$\sigma_\mathrm{sys}$",
    "eta_inst":         r"$\eta_\mathrm{inst}$",
    "eta_coron":        r"$\eta_\mathrm{cor}$",
    "hz_inner_au":      r"$a_\mathrm{HZ,in}$",
    "hz_outer_au":      r"$a_\mathrm{HZ,out}$",
    "radius_factor":    r"$R_p$",
}


def _bar(done, total, t0, width=28):
    frac = done / max(total, 1)
    fill = int(width * frac)
    el = time.time() - t0
    eta = el / max(done, 1) * (total - done)
    return (f"[{done:>5d}/{total} {100*frac:5.1f}%] "
            f"|{'#'*fill}{'-'*(width-fill)}| elapsed {el:5.0f}s eta {eta:5.0f}s")


def _dcol(d):
    return f"yield_D{d:g}"


class Model:
    def __init__(self, cfg, base_dir, max_stars, n_draws, apertures,
                 contrast, kowa, exp_s, mode):
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
        self.apertures = np.asarray(apertures, float)
        self.contrast = contrast
        self.kowa = kowa
        self.exp_s = exp_s
        self.mode = mode
        self.mult = self.n_full / self.frac
        print(f">> model: {len(stars):,} sim stars (of {self.n_full:,} full), "
              f"{n_draws} draws | {len(self.apertures)} apertures "
              f"{self.apertures.tolist()} | contrast={contrast:.0e} k_OWA={kowa:g} "
              f"t={exp_s/60:g}min", flush=True)

    def completeness(self, p):
        """Completeness (per aperture) for parameter set p. Orbits sampled once."""
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
        inst.k_owa = None if self.kowa <= 0 else self.kowa
        bw = (1.0 / self.mode["R"]) if self.mode["name"] == "characterization" \
            else inst.bandwidth

        out = np.empty(len(self.apertures))
        for k, D in enumerate(self.apertures):
            det = photometry.detectability(
                orb, self.g, self.lum, float(D), self.contrast, self.exp_s,
                inst, bandwidth_frac=bw, target_snr=self.mode["snr"])
            out[k] = det["detect"].mean()
        return out

    def yields(self, p, comp=None):
        if comp is None:
            comp = self.completeness(p)
        return p["eta_earth"] * comp * self.mult, comp


# ---------------------------------------------------------------- MC machinery
def draw_params(n_mc, seed):
    rng = np.random.default_rng(seed)
    return {n: rng.uniform(PARAMS[n][1], PARAMS[n][2], size=n_mc) for n in PARAMS}


def evaluate(model, draws, idxs, label=""):
    K = len(model.apertures)
    ys = np.empty((len(idxs), K))
    t0 = time.time()
    step = max(1, len(idxs) // 50)
    for k, i in enumerate(idxs):
        y, _ = model.yields({n: float(draws[n][i]) for n in PARAMS})
        ys[k] = y
        if (k + 1) % step == 0 or k + 1 == len(idxs):
            print(f"   {label}" + _bar(k + 1, len(idxs), t0), flush=True)
    return ys


def _mc_frame(draws, idxs, ys, apertures):
    df = pd.DataFrame({n: draws[n][idxs] for n in PARAMS})
    for k, d in enumerate(apertures):
        df[_dcol(d)] = ys[:, k]
    return df


def run_mc(model, n_mc, seed, out_dir):
    print(f">> global Monte Carlo: {n_mc} draws x {len(model.apertures)} apertures", flush=True)
    draws = draw_params(n_mc, seed)
    idxs = np.arange(n_mc)
    ys = evaluate(model, draws, idxs)
    df = _mc_frame(draws, idxs, ys, model.apertures)
    df.to_csv(os.path.join(out_dir, "mc_samples.csv"), index=False)
    return df


def run_shard(model, n_mc, seed, shard, nshards, tmp_dir):
    draws = draw_params(n_mc, seed)
    idxs = np.arange(n_mc)[shard::nshards]
    print(f">> shard {shard}/{nshards}: {len(idxs)} of {n_mc} draws", flush=True)
    ys = evaluate(model, draws, idxs, label=f"shard{shard} ")
    os.makedirs(tmp_dir, exist_ok=True)
    part = os.path.join(tmp_dir, f"part_{shard:04d}.npz")
    np.savez(part, idx=idxs, y=ys, **{n: draws[n][idxs] for n in PARAMS})
    print(f">> wrote {part}", flush=True)


def combine_shards(tmp_dir, apertures, out_dir):
    files = sorted(glob.glob(os.path.join(tmp_dir, "part_*.npz")))
    if not files:
        raise SystemExit(f"no part_*.npz in {tmp_dir}")
    print(f">> combining {len(files)} MC shards", flush=True)
    idx, ys = [], []
    cols = {n: [] for n in PARAMS}
    for f in files:
        d = np.load(f)
        idx.append(d["idx"]); ys.append(d["y"])
        for n in PARAMS:
            cols[n].append(d[n])
    idx = np.concatenate(idx); order = np.argsort(idx)
    ys = np.concatenate(ys, axis=0)[order]
    draws = {n: np.concatenate(cols[n])[order] for n in PARAMS}
    df = _mc_frame(draws, np.arange(len(idx)), ys, apertures)
    df.to_csv(os.path.join(out_dir, "mc_samples.csv"), index=False)
    return df


def run_oat(model, out_dir):
    print(">> OAT sensitivity (per aperture) ...", flush=True)
    Y0, _ = model.yields(BASELINE)                     # (K,)
    rows = []
    for i, (name, (base, lo, hi)) in enumerate(PARAMS.items(), 1):
        Ylo, _ = model.yields({**BASELINE, name: lo})
        Yhi, _ = model.yields({**BASELINE, name: hi})
        for k, D in enumerate(model.apertures):
            rows.append(dict(parameter=name, diameter=float(D), baseline=base,
                             low=lo, high=hi, Y0=Y0[k], Y_low=Ylo[k],
                             Y_high=Yhi[k], abs_range=abs(Yhi[k] - Ylo[k]),
                             swing_low_pct=100*(Ylo[k]-Y0[k])/max(Y0[k], 1e-9),
                             swing_high_pct=100*(Yhi[k]-Y0[k])/max(Y0[k], 1e-9)))
        print(f"   [{i}/{len(PARAMS)}] {name}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "oat_all.csv"), index=False)
    return df


# ------------------------------------------------------------------- plotting
def _apertures_from(mc_df):
    ds = []
    for c in mc_df.columns:
        if c.startswith("yield_D"):
            ds.append(float(c[len("yield_D"):]))
    return sorted(ds)


def make_plots(mc_df, oat_df, out_dir, ref_ap):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    # column-width figures, fonts matched to the other paper figures
    plt.rcParams.update({"axes.titlesize": 10, "axes.labelsize": 11,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 8})
    COL = (3.7, 3.27)         # single-column size, aspect matched to cube_error figs

    aps = _apertures_from(mc_df)
    ind_dir = os.path.join(out_dir, "individual")
    os.makedirs(ind_dir, exist_ok=True)
    cmap = plt.cm.viridis
    norm = LogNorm(vmin=min(aps), vmax=max(aps))
    eta = mc_df["eta_earth"].to_numpy()

    # per-aperture stats
    stats = {}
    for d in aps:
        y = mc_df[_dcol(d)].to_numpy()
        z = y / eta                                    # eta_Earth removed
        stats[d] = dict(
            median=np.median(y), p16=np.percentile(y, 16), p84=np.percentile(y, 84),
            p2p5=np.percentile(y, 2.5), p97p5=np.percentile(y, 97.5),
            ymin=y.min(), ymax=y.max(),
            mean=y.mean(), std=y.std(),
            rel=y.std()/max(y.mean(), 1e-9), rel_noeta=z.std()/max(z.mean(), 1e-9))

    # ---------- COLLECTIVE 1: overlaid line-histograms (log yield axis) ----------
    # yields span orders of magnitude across apertures, so histogram in log-space
    # on shared bins -> each aperture is a comparable bump that marches rightward.
    allv = mc_df[[_dcol(d) for d in aps]].to_numpy().ravel()
    allv = allv[allv > 0]
    ledges = np.linspace(np.log10(np.percentile(allv, 0.2)),
                         np.log10(allv.max()), 55)
    lctr = 10 ** (0.5 * (ledges[:-1] + ledges[1:]))
    fig, ax = plt.subplots(figsize=(4.1, 3.0))
    for d in aps:
        y = mc_df[_dcol(d)].to_numpy()
        y = y[y > 0]
        c, _ = np.histogram(np.log10(y), bins=ledges, density=True)
        ax.plot(lctr, c, color=cmap(norm(d)), lw=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("expected yield  (log scale)"); ax.set_ylabel("probability density")
    ax.set_title("Yield distribution per aperture")
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array(np.asarray(aps, float))
    cb = fig.colorbar(sm, ax=ax); cb.set_label("aperture D [m]", fontsize=8)
    cb.set_ticks(aps); cb.set_ticklabels([f"{d:g}" for d in aps])
    cb.ax.tick_params(labelsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "mc_hist_overlay.png"),
                                    dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------- COLLECTIVE 2: sensitivity vs diameter (collective tornado) ----------
    fig, ax = plt.subplots(figsize=COL)
    pcmap = plt.cm.tab10(np.linspace(0, 1, len(PARAMS)))
    for c, name in zip(pcmap, PARAMS):
        s = oat_df[oat_df["parameter"] == name].sort_values("diameter")
        ax.plot(s["diameter"].to_numpy(), s["abs_range"].to_numpy(), "-o",
                color=c, ms=3, lw=1.5, label=LABELS.get(name, name))
    ax.set_yscale("log")                                # linear aperture axis
    ax.set_xlabel("aperture D [m]"); ax.set_ylabel("|yield swing|")
    ax.set_title("Parameter sensitivity vs aperture")
    ax.grid(True, which="major", alpha=0.2)             # per-decade gridlines only
    ax.legend(fontsize=7.5, ncol=3, loc="upper left", handlelength=1.1,
              columnspacing=0.8, labelspacing=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "oat_vs_diameter.png"),
                                    dpi=200)
    plt.close(fig)

    # ---------- COLLECTIVE 3: yield band vs aperture ----------
    fig, ax = plt.subplots(figsize=COL)
    flr = lambda v: max(v, 0.1)                        # keep log axis finite
    med = [stats[d]["median"] for d in aps]
    # faint full range (max deviation either side of the median)
    ax.fill_between(aps, [flr(stats[d]["ymin"]) for d in aps],
                    [stats[d]["ymax"] for d in aps], color="#1f77b4", alpha=0.06,
                    label="full range")
    ax.fill_between(aps, [flr(stats[d]["p2p5"]) for d in aps],
                    [stats[d]["p97p5"] for d in aps], color="#1f77b4", alpha=0.16,
                    label="2$\\sigma$")
    ax.fill_between(aps, [flr(stats[d]["p16"]) for d in aps],
                    [stats[d]["p84"] for d in aps], color="#1f77b4", alpha=0.35,
                    label="1$\\sigma$")
    ax.plot(aps, med, "o-", color="#08306b", lw=1.8, ms=3.5, label="median")
    ax.set_yscale("log"); ax.set_xlabel("aperture D [m]")   # linear aperture axis
    ax.set_ylabel("expected yield")
    ax.set_title("Yield variation from parameter uncertainty")
    ax.grid(True, which="major", alpha=0.2)                # major (per-decade) only
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "band_vs_aperture.png"),
                                    dpi=200)
    plt.close(fig)

    # ---------- COLLECTIVE 4: relative systematic vs aperture ----------
    fig, ax = plt.subplots(figsize=COL)
    yv = [100*stats[d]["rel_noeta"] for d in aps]
    ax.plot(aps, yv, "s--", color="#F5821F", lw=1.8, ms=4)
    # value boxes with an orange leader line pointing at each data point
    off = {10: (-4, 26), 100: (24, 22), 250: (26, 20),
           500: (0, 26), 750: (0, 26), 1000: (-10, 26)}
    for d, v in zip(aps, yv):
        if d in off:
            ax.annotate(f"{v:.1f}%", xy=(d, v), xytext=off[d],
                        textcoords="offset points", ha="center", fontsize=7,
                        color="#9a4a00",
                        arrowprops=dict(arrowstyle="-", color="#F5821F", lw=0.9),
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec="#F5821F", lw=0.9, alpha=0.95))
    ax.set_xlabel("aperture D [m]")                     # linear aperture axis
    ax.set_ylabel("best 1$\\sigma$ uncertainty on $\\eta_\\oplus$  "
                  "[$\\pm$% of $\\eta_\\oplus$]")
    ax.set_title("$\\eta_\\oplus$ precision vs aperture")
    ax.set_ylim(0, max(yv) * 1.28)
    ax.grid(True, which="major", alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "relsigma_vs_diameter.png"),
                                    dpi=200)
    plt.close(fig)

    # ---------- INDIVIDUAL per aperture ----------
    for d in aps:
        # tornado
        s = oat_df[oat_df["diameter"] == d].sort_values("abs_range")
        y0 = float(s["Y0"].iloc[0])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for k, (_, r) in enumerate(s.iterrows()):
            ax.barh(k, r["Y_high"] - r["Y_low"], left=min(r["Y_low"], r["Y_high"]),
                    color="#F5821F", alpha=0.85)
        ax.axvline(y0, color="black", ls="--", lw=1, label=f"baseline = {y0:.1f}")
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels([LABELS.get(pp, pp) for pp in s["parameter"]])
        ax.set_xlabel("expected yield"); ax.legend()
        ax.set_title(f"OAT tornado  —  D = {d:g} m")
        fig.tight_layout(); fig.savefig(os.path.join(ind_dir, f"tornado_D{d:g}.png"), dpi=160)
        plt.close(fig)
        # histogram
        fig, ax = plt.subplots(figsize=(7, 4.2))
        y = mc_df[_dcol(d)].to_numpy()
        ax.hist(y, bins=40, color=cmap(norm(d)), alpha=0.85)
        for q, ls in [(16, ":"), (50, "-"), (84, ":")]:
            ax.axvline(np.percentile(y, q), color="black", ls=ls, lw=1.2)
        ax.set_xlabel("expected yield"); ax.set_ylabel("MC draws")
        ax.set_title(f"parameter Monte Carlo  —  D = {d:g} m")
        fig.tight_layout(); fig.savefig(os.path.join(ind_dir, f"mc_hist_D{d:g}.png"), dpi=160)
        plt.close(fig)

    print(f">> wrote collective figures + {len(aps)} individual tornados/hists", flush=True)
    return stats


def write_summary(stats, design, mode, mc_df, out_dir):
    aps = _apertures_from(mc_df)
    summary = dict(design=design, mode=mode, n_mc=int(len(mc_df)),
                   apertures=aps, per_aperture={})
    for d in aps:
        s = stats[d]
        summary["per_aperture"][f"{d:g}"] = dict(
            median=s["median"], p16=s["p16"], p84=s["p84"],
            p2p5=s["p2p5"], p97p5=s["p97p5"],
            rel_sigma=s["rel"], rel_sigma_eta_removed=s["rel_noeta"])
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apertures", type=float, nargs="+",
                   default=[10, 20, 50, 100, 250, 500, 750, 1000])
    p.add_argument("--contrast", type=float, default=1e-10)
    p.add_argument("--kowa", type=float, default=32.0)
    p.add_argument("--exp_min", type=float, default=360.0)
    p.add_argument("--mode", choices=["detection", "characterization"], default="detection")
    p.add_argument("--spectral_R", type=float, default=70.0)
    p.add_argument("--snr", type=float, default=10.0)
    p.add_argument("--n_mc", type=int, default=1500)
    p.add_argument("--max_stars", type=int, default=30000)
    p.add_argument("--n_draws", type=int, default=600)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--ref_aperture", type=float, default=100.0)
    p.add_argument("--out_dir", default="results/sensitivity")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--plots_only", action="store_true",
                   help="regenerate figures from existing mc_samples.csv + oat_all.csv")
    # parallel
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--tmp", default="parts_sens")
    p.add_argument("--combine", action="store_true")
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = args.tmp if os.path.isabs(args.tmp) else os.path.join(base_dir, args.tmp)

    # ---- plots-only: no catalog / no compute ----
    if args.plots_only:
        mc_df = pd.read_csv(os.path.join(out_dir, "mc_samples.csv"))
        oat_df = pd.read_csv(os.path.join(out_dir, "oat_all.csv"))
        stats = make_plots(mc_df, oat_df, out_dir, args.ref_aperture)
        write_summary(stats, {}, args.mode, mc_df, out_dir)
        print(">> plots regenerated.")
        return

    cfg = RunConfig()
    mode = dict(name=args.mode, R=args.spectral_R, snr=args.snr)
    model = Model(cfg, base_dir, args.max_stars, args.n_draws, args.apertures,
                  args.contrast, args.kowa, args.exp_min * MIN, mode)

    # ---- shard mode ----
    if args.nshards > 1 and not args.combine:
        run_shard(model, args.n_mc, args.seed, args.shard, args.nshards, tmp_dir)
        return

    t0 = time.time()
    if args.combine:
        mc_df = combine_shards(tmp_dir, model.apertures, out_dir)
        oat_df = run_oat(model, out_dir)
    else:
        oat_df = run_oat(model, out_dir)
        mc_df = run_mc(model, args.n_mc, args.seed, out_dir)

    design = dict(apertures=model.apertures.tolist(), contrast=args.contrast,
                  kowa=args.kowa, exp_min=args.exp_min)
    if not args.no_plots:
        stats = make_plots(mc_df, oat_df, out_dir, args.ref_aperture)
        write_summary(stats, design, args.mode, mc_df, out_dir)

    print(f"\n>> DONE in {time.time()-t0:.0f}s")
    for d in _apertures_from(mc_df):
        y = mc_df[_dcol(d)].to_numpy()
        print(f"   D={d:>6g} m:  median {np.median(y):7.2f}  "
              f"68% [{np.percentile(y,16):7.2f}, {np.percentile(y,84):7.2f}]  "
              f"(+/-{100*y.std()/max(y.mean(),1e-9):.0f}%)")


if __name__ == "__main__":
    main()
