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
from matplotlib.colors import LinearSegmentedColormap

# ICRS (ra,dec) -> Galactic (l,b) rotation matrix (J2000, standard constants).
_A_G = np.radians(192.85948)   # RA of north galactic pole
_D_G = np.radians(27.12825)    # Dec of north galactic pole
_L_NCP = np.radians(122.93192)  # galactic longitude of the north celestial pole

R0_PC = 8178.0   # Sun -> Galactic centre distance [pc] (GRAVITY 2019); the centre
#                  sits at (X, Y, Z) = (+R0, 0, 0) in our Sun-at-origin frame.


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


def _panel(ax, h, v, lim, survey_pc, style, hlabel, vlabel, gc_arrow,
           fs, star_s, sun_s):
    dark = (style == "dark")
    ink = "white" if dark else "black"
    grey = "#cbd0d6" if dark else "#5a6068"
    ring = "#ffd60a"
    if dark:
        ax.set_facecolor("none")
    # star field: small, faint; alpha scales down as density rises
    ax.scatter(h, v, s=star_s, c=grey, alpha=0.18, linewidths=0, rasterized=True)
    # 300 pc survey sphere (projects to a circle in every plane)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.fill(survey_pc * np.cos(th), survey_pc * np.sin(th),
            color=ring, alpha=0.16, zorder=4)
    ax.plot(survey_pc * np.cos(th), survey_pc * np.sin(th),
            color=ring, lw=1.3, zorder=5)
    # Sun
    ax.scatter([0], [0], s=sun_s, marker="*", c=ring, edgecolors=ink,
               linewidths=0.5, zorder=6)
    if gc_arrow:
        ax.annotate("", xy=(0.92 * lim, 0), xytext=(0.55 * lim, 0),
                    arrowprops=dict(arrowstyle="-|>", color=ink, lw=1.3), zorder=6)
        ax.text(0.90 * lim, 0.04 * lim, "Galactic\ncentre", color=ink,
                fontsize=fs["arrow"], ha="right", va="bottom")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(hlabel, color=ink, fontsize=fs["lab"])
    ax.set_ylabel(vlabel, color=ink, fontsize=fs["lab"])
    ax.tick_params(colors=ink, labelsize=fs["tick"])
    for s in ax.spines.values():
        s.set_color(ink if not dark else (1, 1, 1, 0.5))
    ax.grid(True, color=ink, alpha=0.08, lw=0.5)


def make_figure(df, survey_pc, lim, style, dpi, out_dir, layout="wide"):
    dark = (style == "dark")
    ink = "white" if dark else "black"
    dist = 1000.0 / df["parallax"].to_numpy()
    x, y, z = radec_to_galactic_xyz(df["ra"].to_numpy(),
                                    df["dec"].to_numpy(), dist)
    keep = dist <= lim * np.sqrt(3)          # anything that can land in the box
    x, y, z = x[keep], y[keep], z[keep]

    column = (layout == "column")
    if column:
        # single-column figure: stack the two views vertically, small type
        fig, (axA, axB) = plt.subplots(2, 1, figsize=(3.4, 6.3), dpi=dpi)
        fs = {"lab": 6.5, "tick": 5.5, "arrow": 6.0, "ptitle": 7.5,
              "suptitle": 8.0, "cap": 5.5}
        star_s, sun_s = 0.25, 45
    else:
        # full-width figure: two views side by side
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 5.6), dpi=dpi)
        fs = {"lab": 9, "tick": 7, "arrow": 8, "ptitle": 11,
              "suptitle": 12.5, "cap": 8}
        star_s, sun_s = 0.35, 70

    if dark:
        fig.patch.set_alpha(0.0)
    _panel(axA, x, y, lim, survey_pc, style,
           "X  [pc]  (toward gal. centre)", "Y  [pc]  (toward rotation)",
           True, fs, star_s, sun_s)
    axA.set_title("face-on  (top-down on the disk)", color=ink, fontsize=fs["ptitle"])
    _panel(axB, x, z, lim, survey_pc, style,
           "X  [pc]  (toward gal. centre)", "Z  [pc]  (toward NGP)",
           False, fs, star_s, sun_s)
    axB.set_title("edge-on  (the thin disk from the side)", color=ink,
                  fontsize=fs["ptitle"])

    if column:
        fig.suptitle("The Sun in the local Milky Way disk\n(Gaia DR3; gold ring = "
                     f"{survey_pc:.0f} pc survey volume)",
                     color=ink, fontsize=fs["suptitle"], y=0.995, linespacing=1.2)
        fig.subplots_adjust(left=0.14, right=0.97, top=0.90, bottom=0.075,
                            hspace=0.30)
    else:
        fig.suptitle(f"The Sun in the local Milky Way disk (Gaia DR3)  —  "
                     f"gold ring = {survey_pc:.0f} pc direct-imaging survey volume",
                     color=ink, fontsize=fs["suptitle"], y=0.99)
        fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.11, wspace=0.22)

    os.makedirs(out_dir, exist_ok=True)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    from PIL import Image
    suffix = "_column" if column else ""
    out = os.path.join(out_dir, f"galaxy_context{suffix}_{style}.png")
    if dark:
        Image.fromarray(buf).save(out)
    else:
        Image.fromarray(buf[..., :3]).save(out)
    print(f">> wrote {out}", flush=True)
    return out


# --------------------------------------------------------------------------
# Galaxy-scale "you are here" map: real Gaia bubble + SCHEMATIC Milky Way
# --------------------------------------------------------------------------
def _mw_density(H, V, cx, proj):
    """Schematic Milky Way surface density on a grid (normalised to 1).
    proj='xy' = face-on (spiral disk + bulge); proj='xz' = edge-on (thin slab)."""
    Rd, hz, Rmax = 2600.0, 300.0, 15000.0        # disk scale-length, height, edge
    bulge_amp, bulge_sc = 1.8, 900.0
    if proj == "xy":
        R = np.hypot(H - cx, V)
        phi = np.arctan2(V, H - cx)
        disk = np.exp(-R / Rd)
        # two log-spiral arms (pitch ~12 deg); arm ridges where cos(psi)=1
        pitch = np.radians(12.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            psi = 2 * phi - np.log(np.clip(R, 1.0, None) / 3000.0) / np.tan(pitch)
        arm = np.exp(1.6 * (np.cos(psi) - 1.0))
        body = disk * (1.0 + 1.3 * arm)
        bulge = bulge_amp * np.exp(-R / bulge_sc)
    else:                                          # edge-on
        Rp = np.abs(H - cx)
        rr = np.hypot(H - cx, V)
        R = Rp
        body = np.exp(-Rp / Rd) * np.exp(-np.abs(V) / hz)
        bulge = bulge_amp * np.exp(-rr / bulge_sc)
    edge = 0.5 * (1.0 - np.tanh((R - Rmax) / 1500.0))   # soft disk truncation
    dens = (body + bulge) * edge
    dens = dens / dens.max()
    return dens ** 0.42                              # compress: show the whole disk


def _mw_cmap(style):
    b = (0.29, 0.56, 0.98)                         # milky-way blue
    a_max = 0.75 if style == "dark" else 0.62
    return LinearSegmentedColormap.from_list(
        "mw", [(*b, 0.0), (*b, a_max)])


def _galaxy_panel(ax, H, V, cx, half, zhalf, proj, style, fs,
                  real_h, real_v, survey_pc, real_s, mark_gc_label, xcenter=None):
    dark = (style == "dark")
    ink = "white" if dark else "black"
    real_c = "#e8ecf1" if dark else "#4a5058"      # real stars: white / grey
    orange = "#ff8c00"
    vh = zhalf if proj == "xz" else half
    xc = cx if xcenter is None else xcenter         # frame centre on X
    if dark:
        ax.set_facecolor("none")
    # --- schematic galaxy (blue, low alpha), full disk visible ---
    gh = np.linspace(xc - half, xc + half, 500)
    gv = np.linspace(-vh, vh, 500)
    GH, GV = np.meshgrid(gh, gv)
    dens = _mw_density(GH, GV, cx, proj)
    ax.contourf(GH, GV, dens, levels=np.linspace(0.035, 1.0, 10),
                cmap=_mw_cmap(style), antialiased=True, zorder=1)
    # --- real Gaia bubble (what we actually measured), grey/white, faint ---
    ax.scatter(real_h, real_v, s=real_s, c=real_c, alpha=0.09,
               linewidths=0, rasterized=True, zorder=3)
    # --- "our visual range": the telescope survey volume, as an orange glow ---
    for rr, aa in [(survey_pc * 2.4, 0.10), (survey_pc * 1.7, 0.18),
                   (survey_pc * 1.2, 0.30), (survey_pc, 0.55)]:
        ax.add_patch(plt.Circle((0, 0), rr, color=orange, alpha=aa,
                                lw=0, zorder=4))
    # --- Galactic centre marker (now in frame) ---
    ax.scatter([cx], [0], s=80, marker="o", c="#ffd60a", edgecolors=ink,
               linewidths=0.8, zorder=6)
    # --- Sun marker ---
    ax.scatter([0], [0], s=45, marker="*", c=orange, edgecolors=ink,
               linewidths=0.4, zorder=7)
    # --- compact labels: below the markers, inside the frame, no title clash ---
    ax.annotate("Sun", xy=(0, 0), xytext=(0, -0.34 * vh), textcoords="data",
                color=orange, fontsize=fs["arrow"], ha="center", va="top",
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=orange, lw=0.9), zorder=7)
    if mark_gc_label:
        ax.annotate("Galactic\ncentre", xy=(cx, 0), xytext=(cx, -0.34 * vh),
                    textcoords="data", color=ink, fontsize=fs["arrow"],
                    ha="center", va="top", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=ink, lw=0.9), zorder=6)
    ax.set_xlim(xc - half, xc + half)
    ax.set_ylim(-vh, vh)
    ax.set_aspect("equal")
    ax.tick_params(colors=ink, labelsize=fs["tick"])
    for s in ax.spines.values():
        s.set_color(ink if not dark else (1, 1, 1, 0.5))
    ax.grid(True, color=ink, alpha=0.08, lw=0.5)


def _galaxy_legend(ax, style, loc="upper left", fontsize=8):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    dark = (style == "dark")
    ink = "white" if dark else "black"
    real_c = "#e8ecf1" if dark else "#4a5058"
    blue = (0.29, 0.56, 0.98)
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=6,
               markerfacecolor=real_c, markeredgecolor="none",
               label="Gaia data point"),
        Patch(facecolor=(*blue, 0.55), edgecolor="none",
              label="projected Milky Way region"),
        Patch(facecolor="#ff8c00", edgecolor="none",
              label="range considered"),
    ]
    leg = ax.legend(handles=handles, loc=loc, fontsize=fontsize, framealpha=0.55,
                    facecolor="#0b0f14" if dark else "white",
                    edgecolor=(1, 1, 1, 0.3) if dark else (0, 0, 0, 0.3),
                    labelcolor=ink, handlelength=1.4, borderpad=0.6,
                    labelspacing=0.5)
    leg.set_zorder(10)


def _save_fig(fig, dark, out):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    from PIL import Image
    if dark:
        Image.fromarray(buf).save(out)
    else:
        Image.fromarray(buf[..., :3]).save(out)
    print(f">> wrote {out}", flush=True)


def make_galaxy_map(df, survey_pc, style, dpi, out_dir,
                    gal_half=16000.0, real_max=3000.0):
    """Two SEPARATE galaxy-scale figures (face-on and edge-on): the real Gaia
    bubble overlaid on a schematic Milky Way (blue)."""
    dark = (style == "dark")
    ink = "white" if dark else "black"
    cx = R0_PC
    zhalf = 6000.0
    dist = 1000.0 / df["parallax"].to_numpy()
    x, y, z = radec_to_galactic_xyz(df["ra"].to_numpy(),
                                    df["dec"].to_numpy(), dist)
    keep = dist <= real_max
    x, y, z = x[keep], y[keep], z[keep]
    os.makedirs(out_dir, exist_ok=True)

    # The face-on figure is placed at ~half the text width in the paper, so it is
    # scaled down more than the (near-full-width) edge-on one. To make the two
    # match in PRINT, the face-on uses ~1.25x larger source fonts; the edge-on
    # drops a couple of points. Titles are axes-attached (tight) to kill the gap.
    edge_fs = {"tick": 6.5, "arrow": 7.5, "title": 10.5, "lab": 7.5, "leg": 6.5}
    face_fs = {"tick": 8.5, "arrow": 9.5, "title": 13.0, "lab": 9.5, "leg": 8.5}

    # ---- face-on figure (square) ----
    fig, ax = plt.subplots(figsize=(5.6, 5.8), dpi=dpi)
    if dark:
        fig.patch.set_alpha(0.0)
    _galaxy_panel(ax, x, y, cx, gal_half, zhalf, "xy", style, face_fs,
                  x, y, survey_pc, 1.0, mark_gc_label=True, xcenter=0.0)
    _galaxy_legend(ax, style, loc="upper left", fontsize=face_fs["leg"])
    ax.set_xlabel("X  [pc]  (toward galactic centre)", color=ink, fontsize=face_fs["lab"])
    ax.set_ylabel("Y  [pc]", color=ink, fontsize=face_fs["lab"])
    ax.set_title("Observation area relative to Milky Way — face-on",
                 color=ink, fontsize=face_fs["title"], pad=5)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.955, bottom=0.085)
    _save_fig(fig, dark, os.path.join(out_dir, f"galaxy_map_faceon_{style}.png"))

    # ---- edge-on figure (wide, short) ----
    fig, ax = plt.subplots(figsize=(9.0, 3.9), dpi=dpi)
    if dark:
        fig.patch.set_alpha(0.0)
    _galaxy_panel(ax, x, z, cx, gal_half, zhalf, "xz", style, edge_fs,
                  x, z, survey_pc, 1.0, mark_gc_label=True)
    _galaxy_legend(ax, style, loc="upper left", fontsize=edge_fs["leg"])
    ax.set_xlabel("X  [pc]  (toward galactic centre)", color=ink, fontsize=edge_fs["lab"])
    ax.set_ylabel("Z  [pc]", color=ink, fontsize=edge_fs["lab"])
    ax.set_title("Observation area relative to Milky Way — edge-on", color=ink,
                 fontsize=edge_fs["title"], pad=5)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.155)
    _save_fig(fig, dark, os.path.join(out_dir, f"galaxy_map_edgeon_{style}.png"))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default="gaia_galaxy_context.csv")
    p.add_argument("--survey_pc", type=float, default=300.0)
    p.add_argument("--lim_pc", type=float, default=3000.0,
                   help="plot half-width [pc]")
    p.add_argument("--style", choices=["paper", "dark", "both"], default="both")
    p.add_argument("--layout", choices=["wide", "column", "both"], default="column",
                   help="column = single-column (stacked, ~3.4in wide); "
                        "wide = full-width two-panel")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--out_dir", default="results/galaxy_context")
    p.add_argument("--galaxy", action="store_true",
                   help="make the galaxy-scale map (real Gaia bubble on a blue "
                        "schematic Milky Way) instead of the local 3 kpc plots")
    p.add_argument("--gal_half_pc", type=float, default=16000.0,
                   help="galaxy-map half-width [pc] (default 16 kpc)")
    p.add_argument("--real_max_pc", type=float, default=3000.0,
                   help="clip the real Gaia bubble to this radius [pc]")
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
    layouts = ["wide", "column"] if args.layout == "both" else [args.layout]
    for style in styles:
        for layout in layouts:
            if args.galaxy:
                make_galaxy_map(df, args.survey_pc, style, args.dpi, out_dir,
                                gal_half=args.gal_half_pc,
                                real_max=args.real_max_pc)
                break                      # galaxy map isn't layout-dependent
            else:
                make_figure(df, args.survey_pc, args.lim_pc, style, args.dpi,
                            out_dir, layout=layout)


if __name__ == "__main__":
    main()
