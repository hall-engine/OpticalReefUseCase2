#!/usr/bin/env python3
"""
cloud_grid.py  --  static 3D-cloud grid: rows = aperture diameters, columns =
contrast floors, one figure per integration time.

Each cell is the local stellar cloud (Sun at the origin, within --cap_pc); grey =
all HZ-relevant stars, red = those where an HZ Earth analog is detectable
(p_det > --threshold) at that (diameter, contrast, integration time). OWA is
applied (config k_owa). Reads the catalog (per-star), not the cube.

    python3 cloud_grid.py                       # 5 figures, times 5min..12h
    python3 cloud_grid.py --times_min 60 --style both
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from config import RunConfig
import catalog
import montecarlo
import photometry
from animate_cloud3d import radec_to_xyz


def fmt_time(minutes):
    if minutes < 60:
        return f"{minutes:g} min"
    h = minutes / 60.0
    return f"{h:g} h" if h < 24 else f"{h / 24:g} d"


def _cloud_panel(ax, xyz, mask, style, cap, elev, azim):
    dark = (style == "dark")
    ink = "white" if dark else "black"
    grey = "#c8ccd2" if dark else "#8a9099"
    red = "#ff453a" if dark else "#d62728"
    x, y, z = xyz
    if dark:
        ax.set_facecolor("none")
    ax.scatter(x[~mask], y[~mask], z[~mask], s=1.2, c=grey, alpha=0.25,
               linewidths=0, depthshade=True)
    ax.scatter(x[mask], y[mask], z[mask], s=4, c=red, alpha=0.85,
               linewidths=0, depthshade=True)
    ax.scatter([0], [0], [0], s=55, c="#ffd60a", marker="*",
               edgecolors=ink, linewidths=0.3, depthshade=False)
    ax.set_xlim(-cap, cap); ax.set_ylim(-cap, cap); ax.set_zlim(-cap, cap)
    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.18)   # fill the cell without spilling
    except TypeError:                              # older mpl: no zoom kwarg
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    for pane_axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            pane_axis.set_pane_color((0, 0, 0, 0))
        except Exception:
            pass
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)


def make_grid(xyz, orb, g, lum, inst, bw, snr, thr, diameters, contrasts,
              t_min, style, cap, elev, azim, dpi, out_dir):
    ink = "white" if style == "dark" else "black"
    nR, nC = len(diameters), len(contrasts)
    fig = plt.figure(figsize=(3.2 * nC, 3.0 * nR), dpi=dpi)
    if style == "dark":
        fig.patch.set_alpha(0.0)
    for i, D in enumerate(diameters):
        for j, con in enumerate(contrasts):
            det = photometry.detectability(orb, g, lum, D, con, t_min * 60.0,
                                           inst, bandwidth_frac=bw, target_snr=snr)
            mask = det["detect"].mean(axis=1) > thr
            ax = fig.add_subplot(nR, nC, i * nC + j + 1, projection="3d")
            _cloud_panel(ax, xyz, mask, style, cap, elev, azim)
            pct = 100.0 * mask.mean()
            ax.text2D(0.5, 0.90, f"{pct:4.1f}%", transform=ax.transAxes,
                      ha="center", color=ink, fontsize=9)
            if i == 0:
                ax.text2D(0.5, 1.0, f"contrast {con:.0e}", transform=ax.transAxes,
                          ha="center", color=ink, fontsize=11)
            if j == 0:
                ax.text2D(-0.02, 0.5, f"D = {D:g} m", transform=ax.transAxes,
                          va="center", ha="center", rotation=90, color=ink,
                          fontsize=11)
    fig.suptitle(f"detectable HZ Earth analogs within {cap:.0f} pc  —  "
                 f"integration {fmt_time(t_min)}", color=ink, fontsize=13, y=0.99)
    # trim margins; keep spacing >= 0 so adjacent 3D boxes touch but don't overlap
    fig.subplots_adjust(left=0.04, right=0.995, top=0.95, bottom=0.01,
                        wspace=0.0, hspace=0.02)

    tag = f"{t_min:g}min".replace(".", "p")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)

    from PIL import Image
    out = os.path.join(out_dir, f"cloud_grid_t{tag}_{style}.png")
    if style == "dark":
        Image.fromarray(buf).save(out)          # keep alpha (transparent)
    else:
        Image.fromarray(buf[..., :3]).save(out)
    print(f">> wrote {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--times_min", type=float, nargs="+", default=[5, 30, 120, 360, 720])
    p.add_argument("--diameters", type=float, nargs="+", default=[5, 20, 100, 1000])
    p.add_argument("--contrasts", type=float, nargs="+", default=[1e-9, 1e-10, 1e-11])
    p.add_argument("--cap_pc", type=float, default=300.0)
    p.add_argument("--max_stars", type=int, default=5000)
    p.add_argument("--n_draws", type=int, default=500)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--k_owa", type=float, default=None,
                   help="override OWA k (default: config k_owa=32; 0 disables)")
    p.add_argument("--style", choices=["paper", "dark", "both"], default="paper")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--elev", type=float, default=18.0)
    p.add_argument("--azim", type=float, default=30.0)
    p.add_argument("--out_dir", default="results/cloud_grids")
    p.add_argument("--catalog", default=None)
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cfg = RunConfig()
    if args.catalog:
        cfg.survey.catalog_path = args.catalog
    cfg.survey.dist_max_pc = args.cap_pc
    cfg.montecarlo.n_draws = args.n_draws
    if args.k_owa is not None:
        cfg.instrument.k_owa = None if args.k_owa == 0 else args.k_owa

    rng = np.random.default_rng(cfg.montecarlo.seed)
    raw = catalog.load_catalog(cfg.survey, base_dir=base_dir)
    cfg.survey.test_max_stars = args.max_stars
    stars, _ = catalog.select_stars(raw, cfg.survey, cfg.montecarlo,
                                    test_mode=True, rng=rng)
    orb = montecarlo.sample_orbits(stars["hz_inner_m"].to_numpy(),
                                   stars["hz_outer_m"].to_numpy(),
                                   stars["distance_m"].to_numpy(), cfg.montecarlo, rng=rng)
    g = stars["phot_g_mean_mag"].to_numpy()
    lum = stars["lum_flame"].to_numpy()
    xyz = radec_to_xyz(stars["ra"].to_numpy(), stars["dec"].to_numpy(),
                       stars["distance_pc"].to_numpy())
    inst = cfg.instrument
    bw = inst.bandwidth
    snr = cfg.observation.snr_detect
    print(f">> {len(stars)} stars within {args.cap_pc:.0f} pc | "
          f"{len(args.diameters)}x{len(args.contrasts)} grid | "
          f"{len(args.times_min)} integration times", flush=True)

    styles = ["paper", "dark"] if args.style == "both" else [args.style]
    for t in args.times_min:
        for style in styles:
            make_grid(xyz, orb, g, lum, inst, bw, snr, args.threshold,
                      args.diameters, args.contrasts, t, style, args.cap_pc,
                      args.elev, args.azim, args.dpi, out_dir)


if __name__ == "__main__":
    main()
