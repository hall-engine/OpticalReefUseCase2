#!/usr/bin/env python3
"""
plot_limits.py
==============
"What limits the yield?" decomposition.

Total expected yield vs aperture, with a separate line for each physical
constraint acting ALONE, so you can see which one binds at each aperture:

    IWA  (resolution / diffraction)  -- planet must sit outside k_IWA·lambda/D.
                                        Rises with D (bigger mirror resolves the
                                        HZ of more distant stars).
    OWA  (dark-hole outer edge)      -- planet must sit inside k_OWA·lambda/D,
                                        set by the deformable-mirror actuator
                                        count. FALLS with D (the correctable
                                        annulus shrinks in angle as D grows).
    contrast / SNR                   -- planet must clear the SNR threshold at
                                        the given raw contrast floor. Rises with
                                        collecting area, then saturates.
    ACTUAL                           -- all three together (the binding envelope);
                                        peaks at the sweet-spot aperture.

One panel per integration time, so you can watch the contrast/SNR line lift as
exposure grows while the geometry limits (IWA, OWA) stay put.

Everything is a CLI knob:

    python3 plot_limits.py                              # defaults
    python3 plot_limits.py --contrast 1e-10 --k_owa 32 \
        --times_days 1 10 30 100 --n_apertures 40
    python3 plot_limits.py --mode characterization --spectral_R 70
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RunConfig
import catalog
import montecarlo
import photometry

MIN = 60.0        # seconds per minute


def fmt_time(minutes):
    """Human label for an integration time given in minutes."""
    if minutes < 60:
        return f"{minutes:g} min"
    hours = minutes / 60.0
    if hours < 24:
        return f"{hours:g} h"
    return f"{hours / 24:g} d"


def component_yield(mask, eta, frac):
    """Expected local yield if `mask` (n_stars, n_draws) were the only gate."""
    pdet = mask.mean(axis=1)
    return eta * pdet.sum() / frac


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="results/limits")
    p.add_argument("--contrast", type=float, default=1e-10,
                   help="raw contrast floor for this decomposition")
    p.add_argument("--k_owa", type=float, default=32.0,
                   help="OWA = k_owa·lambda/D (DM actuator format). Set 0 to disable.")
    p.add_argument("--times_min", type=float, nargs="+",
                   default=[5, 15, 30, 60, 120, 240, 480, 720],
                   help="integration times [minutes] (default 5 min–12 h, 8 steps)")
    p.add_argument("--n_apertures", type=int, default=40)
    p.add_argument("--dmin", type=float, default=4.0)
    p.add_argument("--dmax", type=float, default=1000.0)
    p.add_argument("--n_draws", type=int, default=800)
    p.add_argument("--max_stars", type=int, default=3000)
    p.add_argument("--mode", choices=["detection", "characterization"],
                   default="detection")
    p.add_argument("--spectral_R", type=float, default=70.0)
    p.add_argument("--snr", type=float, default=10.0)
    p.add_argument("--metric", choices=["percent", "yield", "completeness"],
                   default="percent",
                   help="percent = %% of viable candidates (default); "
                        "yield = expected count; completeness = fraction")
    p.add_argument("--logy", action="store_true", help="log y-axis")
    p.add_argument("--three", action="store_true",
                   help="also make the 3-contrast side-by-side figure (Option B)")
    p.add_argument("--contrasts", type=float, nargs="+",
                   default=[1e-9, 1e-10, 1e-11],
                   help="contrast floors for the --three panels")
    p.add_argument("--catalog", default=None)
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cfg = RunConfig()
    if args.catalog:
        cfg.survey.catalog_path = args.catalog
    cfg.instrument.k_owa = None if args.k_owa == 0 else args.k_owa
    cfg.montecarlo.n_draws = args.n_draws

    # effective bandwidth for the mode
    if args.mode == "characterization":
        bw_frac = 1.0 / args.spectral_R
        mode_label = f"characterization (R={args.spectral_R:.0f})"
    else:
        bw_frac = cfg.instrument.bandwidth
        mode_label = "detection (broadband)"

    # ---- sample + geometry (once) ----
    rng = np.random.default_rng(cfg.montecarlo.seed)
    raw = catalog.load_catalog(cfg.survey, base_dir=base_dir)
    cfg.survey.test_max_stars = args.max_stars
    stars, n_full = catalog.select_stars(raw, cfg.survey, cfg.montecarlo,
                                         test_mode=bool(args.max_stars), rng=rng)
    orb = montecarlo.sample_orbits(stars["hz_inner_m"].to_numpy(),
                                   stars["hz_outer_m"].to_numpy(),
                                   stars["distance_m"].to_numpy(),
                                   cfg.montecarlo, rng=rng)
    g = stars["phot_g_mean_mag"].to_numpy()
    lum = stars["lum_flame"].to_numpy()
    n_stars = len(stars)
    frac = cfg.survey.rand_fraction or 1.0
    eta = cfg.drake.eta_earth
    cap_pc = cfg.survey.dist_max_pc
    # yield: completeness (mean pdet) -> full selected population (n_full) -> /frac
    norm = eta * n_full / n_stars / frac
    if args.metric == "yield":
        ylabel = f"expected yield (local pop., < {cap_pc:.0f} pc)"
    elif args.metric == "completeness":
        ylabel = "HZ completeness"
    else:  # percent of viable candidates
        ylabel = f"yield [% of viable candidates within {cap_pc:.0f} pc]"
    print(f">> {n_stars} stars | contrast {args.contrast:.0e} | k_owa={args.k_owa} "
          f"| {mode_label}")

    apertures = np.geomspace(args.dmin, args.dmax, args.n_apertures)

    def comp(mask):
        pdet = mask.mean(axis=1)
        if args.metric == "yield":
            return pdet.sum() * norm
        if args.metric == "percent":
            return pdet.mean() * 100.0
        return pdet.mean()

    # ---- compute limit curves ----
    times_s = [t * MIN for t in args.times_min]
    # geometry limits (time-independent): IWA-only, OWA-only
    y_iwa = np.zeros(len(apertures))
    y_owa = np.zeros(len(apertures))
    # per-time: SNR-only and ACTUAL
    y_snr = {t: np.zeros(len(apertures)) for t in args.times_min}
    y_act = {t: np.zeros(len(apertures)) for t in args.times_min}

    for i, D in enumerate(apertures):
        # one call gives the geometry masks (time-independent)
        base = photometry.detectability(orb, g, lum, D, args.contrast, MIN,
                                        cfg.instrument, bandwidth_frac=bw_frac,
                                        target_snr=args.snr)
        y_iwa[i] = comp(base["iwa_ok"])
        y_owa[i] = comp(base["owa_ok"])
        for t, ts in zip(args.times_min, times_s):
            det = photometry.detectability(orb, g, lum, D, args.contrast, ts,
                                           cfg.instrument, bandwidth_frac=bw_frac,
                                           target_snr=args.snr)
            y_snr[t][i] = comp(det["snr_ok"])
            y_act[t][i] = comp(det["iwa_ok"] & det["owa_ok"] & det["snr_ok"])
        print(f"   {i+1}/{len(apertures)}  D={D:7.1f} m", end="\r")
    print()

    # ---- plot: one panel per integration time ----
    n = len(args.times_min)
    ncol = min(n, 2)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4 * ncol, 4.6 * nrow),
                             squeeze=False, sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, t in zip(axes, args.times_min):
        ax.plot(apertures, y_iwa, "--", color="#1f77b4", lw=1.6,
                label="IWA (resolution) limit")
        if cfg.instrument.k_owa is not None:
            ax.plot(apertures, y_owa, "--", color="#9467bd", lw=1.6,
                    label=f"OWA limit (k={args.k_owa:g})")
        ax.plot(apertures, y_snr[t], "--", color="#d62728", lw=1.6,
                label="contrast / SNR limit")
        ax.plot(apertures, y_act[t], "-", color="black", lw=2.6,
                label="ACTUAL (combined)")
        ax.set_xscale("log")
        if args.logy:
            ax.set_yscale("log")
        ax.set_title(f"integration = {fmt_time(t)}", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Aperture diameter D [m]")
    for r in range(nrow):
        axes[r * ncol].set_ylabel(ylabel)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"what limits the yield  —  contrast {args.contrast:.0e}",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(out_dir, "yield_limits.png")
    fig.savefig(out, dpi=190)
    plt.close(fig)
    print(f">> wrote {out}")

    # ---- combined single-panel view: IWA/OWA once, contrast+actual per exposure
    _plot_combined(apertures, y_iwa, y_owa, y_snr, y_act, args, cfg,
                   ylabel, mode_label, out_dir)

    # ---- 3-contrast side-by-side (Option B: IWA/OWA + ACTUAL per exposure) ----
    if args.three:
        _plot_three_contrast(orb, g, lum, cfg, bw_frac, apertures, times_s,
                             y_iwa, y_owa, comp, args, ylabel, mode_label, out_dir)


def _plot_combined(apertures, y_iwa, y_owa, y_snr, y_act, args, cfg,
                   ylabel, mode_label, out_dir):
    """All exposures on ONE rectangular, log-y axes.

    IWA and OWA are time-independent -> one line each. Contrast/SNR and ACTUAL
    are drawn per exposure; integration time is encoded by shade (ACTUAL, matched
    to a colorbar on the right) and by opacity (contrast/SNR). Short = light,
    long = dark/solid.
    """
    from matplotlib.lines import Line2D
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm, LinearSegmentedColormap

    times = sorted(args.times_min)
    tmin, tmax = min(times), max(times)
    c_iwa, c_owa, c_snr = "#1f77b4", "#F5821F", "#d62728"   # OWA = Reef orange
    # ACTUAL shade ramp (light grey -> black), matched to the colorbar
    tcmap = LinearSegmentedColormap.from_list(
        "act", plt.cm.Greys(np.linspace(0.40, 1.0, 256)))
    norm = LogNorm(vmin=tmin, vmax=tmax) if tmax > tmin else None

    def tcolor(t):
        return tcmap(1.0) if norm is None else tcmap(norm(t))

    def talpha(t):
        if tmax == tmin:
            return 1.0
        f = (np.log10(t) - np.log10(tmin)) / (np.log10(tmax) - np.log10(tmin))
        return 0.30 + 0.70 * f

    fig, ax = plt.subplots(figsize=(12.5, 5.2))       # rectangular

    ax.plot(apertures, y_iwa, "--", color=c_iwa, lw=2.0, zorder=3)
    if cfg.instrument.k_owa is not None:
        ax.plot(apertures, y_owa, "--", color=c_owa, lw=2.0, zorder=3)
    for t in times:
        ax.plot(apertures, y_snr[t], "-", color=c_snr, lw=1.5,
                alpha=talpha(t), zorder=4)
        ax.plot(apertures, y_act[t], "-", color=tcolor(t), lw=2.6, zorder=5)

    ax.set_xscale("log")
    ymax = np.nanmax(y_iwa)
    if cfg.instrument.k_owa is not None:
        ymax = max(ymax, np.nanmax(y_owa))
    if args.metric == "percent":
        ax.set_ylim(0, 105)                            # linear for percentages
    else:
        ax.set_yscale("log")
        ax.set_ylim(1.0, ymax * 1.4)
    ax.set_xlabel("Aperture diameter D [m]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"what limits the yield  —  contrast {args.contrast:.0e}",
                 fontsize=12)
    ax.grid(True, which="both", alpha=0.2)

    id_handles = [Line2D([0], [0], color=c_iwa, lw=2, ls="--",
                         label="IWA (resolution)"),
                  Line2D([0], [0], color=c_snr, lw=2, label="contrast / SNR"),
                  Line2D([0], [0], color="black", lw=2.6, label="actual yield")]
    if cfg.instrument.k_owa is not None:
        id_handles.insert(1, Line2D([0], [0], color=c_owa, lw=2, ls="--",
                                    label=f"OWA (k={args.k_owa:g})"))
    ax.legend(handles=id_handles, loc="upper left", fontsize=9, title="limit")

    # colorbar (right) = integration time, matched to the ACTUAL shade ramp
    if norm is not None:
        sm = ScalarMappable(norm=norm, cmap=tcmap)
        sm.set_array(np.asarray(times, dtype=float))
        cbar = fig.colorbar(sm, ax=ax, pad=0.015, aspect=26)
        cbar.set_label("integration time")
        cbar.set_ticks(times)
        cbar.set_ticklabels([fmt_time(t) for t in times])

    fig.tight_layout()
    out = os.path.join(out_dir, "yield_limits_combined.png")
    fig.savefig(out, dpi=190)
    plt.close(fig)
    print(f">> wrote {out}")


def _plot_three_contrast(orb, g, lum, cfg, bw_frac, apertures, times_s,
                         y_iwa, y_owa, comp, args, ylabel, mode_label, out_dir):
    """Three contrast floors side by side (like the 3D-map row). Each panel:
    IWA + OWA (contrast-independent, drawn once) and the ACTUAL yield per
    exposure, shaded by integration time. One shared colorbar on the right."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm, LinearSegmentedColormap
    from matplotlib.lines import Line2D

    contrasts = list(args.contrasts)
    times = sorted(args.times_min)
    tmin, tmax = min(times), max(times)
    c_iwa, c_owa = "#1f77b4", "#F5821F"                 # OWA = Reef orange
    tcmap = LinearSegmentedColormap.from_list(
        "act", plt.cm.Greys(np.linspace(0.40, 1.0, 256)))
    norm = LogNorm(vmin=tmin, vmax=tmax) if tmax > tmin else None

    def tcolor(t):
        return tcmap(1.0) if norm is None else tcmap(norm(t))

    # compute ACTUAL yield per contrast per exposure (IWA/OWA reused)
    print(">> 3-contrast: computing ACTUAL per contrast ...")
    y_act = {c: {t: np.zeros(len(apertures)) for t in times} for c in contrasts}
    for c in contrasts:
        for i, D in enumerate(apertures):
            for t, ts in zip(times, times_s):
                det = photometry.detectability(orb, g, lum, D, c, ts,
                                               cfg.instrument, bandwidth_frac=bw_frac,
                                               target_snr=args.snr)
                y_act[c][t][i] = comp(det["iwa_ok"] & det["owa_ok"] & det["snr_ok"])

    ymax = np.nanmax(y_iwa)
    if cfg.instrument.k_owa is not None:
        ymax = max(ymax, np.nanmax(y_owa))

    fig, axes = plt.subplots(1, len(contrasts), figsize=(5.6 * len(contrasts), 5.4),
                             sharex=True, sharey=True)
    if len(contrasts) == 1:
        axes = [axes]
    for ax, c in zip(axes, contrasts):
        ax.plot(apertures, y_iwa, "--", color=c_iwa, lw=1.8, zorder=3)
        if cfg.instrument.k_owa is not None:
            ax.plot(apertures, y_owa, "--", color=c_owa, lw=1.8, zorder=3)
        for t in times:
            ax.plot(apertures, y_act[c][t], "-", color=tcolor(t), lw=2.4, zorder=5)
        ax.set_xscale("log")
        if args.metric == "percent":
            ax.set_ylim(0, 105)                         # linear for percentages
        else:
            ax.set_yscale("log")
            ax.set_ylim(1.0, ymax * 1.4)
        ax.set_xlabel("Aperture diameter D [m]")
        ax.set_title(f"contrast floor {c:.0e}", fontsize=11)
        ax.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel(ylabel)

    id_handles = [Line2D([0], [0], color=c_iwa, lw=1.8, ls="--", label="IWA"),
                  Line2D([0], [0], color="black", lw=2.4, label="actual yield")]
    if cfg.instrument.k_owa is not None:
        id_handles.insert(1, Line2D([0], [0], color=c_owa, lw=1.8, ls="--",
                                    label=f"OWA (k={args.k_owa:g})"))
    axes[0].legend(handles=id_handles, loc="upper left", fontsize=8.5)

    if norm is not None:
        sm = ScalarMappable(norm=norm, cmap=tcmap)
        sm.set_array(np.asarray(times, dtype=float))
        cbar = fig.colorbar(sm, ax=axes, pad=0.012, aspect=30)
        cbar.set_label("integration time")
        cbar.set_ticks(times)
        cbar.set_ticklabels([fmt_time(t) for t in times])

    fig.suptitle("yield vs aperture across contrast floors", fontsize=13)
    out = os.path.join(out_dir, "yield_limits_3contrast.png")
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f">> wrote {out}")


if __name__ == "__main__":
    main()
