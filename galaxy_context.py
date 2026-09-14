#!/usr/bin/env python3
"""
galaxy_context.py  --  "you are here" context figure from a tiny all-sky Gaia DR3
sample (gaia_galaxy_context.csv from fetch_galaxy_context.py).

Two panels, Sun at the origin, GALACTIC Cartesian coordinates [pc]:
  * left  = face-on  (X-Y): looking down on the disk from the north galactic pole;
            X points toward the galactic centre, Y toward galactic rotation.
  * right = edge-on  (X-Z): the thin disk seen from the side -- the flattened slab.
Grey points are Gaia stars (distance = 1000/parallax). A gold ring + shaded core
marks the 300 pc habitable-survey sphere; the Sun is a gold star at (0,0). An
arrow points toward the galactic centre (l=0).

    python3 galaxy_context.py                       # paper + dark
    python3 galaxy_context.py --style dark --survey_pc 300

Note: Gaia parallaxes only reach a few kpc reliably, so this traces the LOCAL
disk (the thin-disk slab and the Sun's place in it), not the whole Galaxy.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ICRS (ra,dec) -> Galactic (l,b) rotation matrix (J2000, standard constants).
_A_G = np.radians(192.85948)   # RA of north galactic pole
_D_G = np.radians(27.12825)    # Dec of north galactic pole
_L_NCP = np.radians(122.93192)  # galactic longitude of the north celestial pole


def radec_to_galactic_xyz(ra_deg, dec_deg, dist_pc):
    """ICRS ra/dec [deg] + distance [pc] -> galactic Cartesian (x,y,z) [pc]."""
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    sb = (np.sin(_D_G) * np.sin(dec)
          + np.cos(_D_G) * np.cos(dec) * np.cos(ra - _A_G))
    b = np.arcsin(np.clip(sb, -1, 1))
    y_ = np.cos(dec) * np.sin(ra - _A_G)
    x_ = (np.cos(_D_G) * np.sin(dec)
          - np.sin(_D_G) * np.cos(dec) * np.cos(ra - _A_G))
    l = _L_NCP - np.arctan2(y_, x_)
    l = np.mod(l, 2 * np.pi)
    x = dist_pc * np.cos(b) * np.cos(l)   # toward galactic centre (l=0)
    y = dist_pc * np.cos(b) * np.sin(l)   # toward galactic rotation (l=90)
    z = dist_pc * np.sin(b)               # toward north galactic pole
    return x, y, z


def _panel(ax, h, v, lim, survey_pc, style, hlabel, vlabel, gc_arrow):
    dark = (style == "dark")
    ink = "white" if dark else "black"
    grey = "#cbd0d6" if dark else "#5a6068"
    ring = "#ffd60a"
    if dark:
        ax.set_facecolor("none")
    # star field: small, faint; alpha scales down as density rises
    ax.scatter(h, v, s=0.35, c=grey, alpha=0.18, linewidths=0, rasterized=True)
    # 300 pc survey sphere (projects to a circle in every plane)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.fill(survey_pc * np.cos(th), survey_pc * np.sin(th),
            color=ring, alpha=0.16, zorder=4)
    ax.plot(survey_pc * np.cos(th), survey_pc * np.sin(th),
            color=ring, lw=1.4, zorder=5)
    # Sun
    ax.scatter([0], [0], s=70, marker="*", c=ring, edgecolors=ink,
               linewidths=0.5, zorder=6)
    if gc_arrow:
        ax.annotate("", xy=(0.92 * lim, 0), xytext=(0.55 * lim, 0),
                    arrowprops=dict(arrowstyle="-|>", color=ink, lw=1.4), zorder=6)
        ax.text(0.93 * lim, 0.04 * lim, "Galactic\ncentre", color=ink,
                fontsize=8, ha="right", va="bottom")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(hlabel, color=ink, fontsize=9)
    ax.set_ylabel(vlabel, color=ink, fontsize=9)
    ax.tick_params(colors=ink, labelsize=7)
    for s in ax.spines.values():
        s.set_color(ink if not dark else (1, 1, 1, 0.5))
    ax.grid(True, color=ink, alpha=0.08, lw=0.5)


def make_figure(df, survey_pc, lim, style, dpi, out_dir):
    dark = (style == "dark")
    ink = "white" if dark else "black"
    dist = 1000.0 / df["parallax"].to_numpy()
    x, y, z = radec_to_galactic_xyz(df["ra"].to_numpy(),
                                    df["dec"].to_numpy(), dist)
    keep = dist <= lim * np.sqrt(3)          # anything that can land in the box
    x, y, z = x[keep], y[keep], z[keep]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.6), dpi=dpi)
    if dark:
        fig.patch.set_alpha(0.0)
    _panel(axL, x, y, lim, survey_pc, style,
           "X  [pc]  (toward galactic centre)", "Y  [pc]  (toward rotation)",
           gc_arrow=True)
    axL.set_title("face-on  (top-down on the disk)", color=ink, fontsize=11)
    _panel(axR, x, z, lim, survey_pc, style,
           "X  [pc]  (toward galactic centre)", "Z  [pc]  (toward NGP)",
           gc_arrow=False)
    axR.set_title("edge-on  (the thin disk from the side)", color=ink, fontsize=11)

    fig.suptitle(f"The Sun in the local Milky Way disk (Gaia DR3)  —  "
                 f"gold ring = {survey_pc:.0f} pc direct-imaging survey volume",
                 color=ink, fontsize=12.5, y=0.99)
    fig.text(0.5, 0.015,
             f"{len(x):,} stars  |  distance = 1000/parallax  |  "
             f"box half-width {lim:.0f} pc", color=ink, ha="center", fontsize=8,
             alpha=0.8)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.11, wspace=0.22)

    os.makedirs(out_dir, exist_ok=True)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    from PIL import Image
    out = os.path.join(out_dir, f"galaxy_context_{style}.png")
    if dark:
        Image.fromarray(buf).save(out)
    else:
        Image.fromarray(buf[..., :3]).save(out)
    print(f">> wrote {out}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default="gaia_galaxy_context.csv")
    p.add_argument("--survey_pc", type=float, default=300.0)
    p.add_argument("--lim_pc", type=float, default=3000.0,
                   help="plot half-width [pc]")
    p.add_argument("--style", choices=["paper", "dark", "both"], default="both")
    p.add_argument("--dpi", type=int, default=170)
    p.add_argument("--out_dir", default="results/galaxy_context")
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    cat_path = args.catalog
    if not os.path.isabs(cat_path):
        cat_path = os.path.join(base_dir, cat_path)
    out_dir = os.path.join(base_dir, args.out_dir)
    df = pd.read_csv(cat_path)
    df = df[df["parallax"] > 0]
    print(f">> {len(df):,} stars from {os.path.basename(cat_path)}", flush=True)

    styles = ["paper", "dark"] if args.style == "both" else [args.style]
    for style in styles:
        make_figure(df, args.survey_pc, args.lim_pc, style, args.dpi, out_dir)


if __name__ == "__main__":
    main()
