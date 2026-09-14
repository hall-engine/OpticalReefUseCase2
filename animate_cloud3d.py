#!/usr/bin/env python3
"""
animate_cloud3d.py
==================
Animated 3D map of the local stellar neighbourhood, Sun at the origin.

Three 3D panels side by side, one per contrast floor. In each:
  * Sun    = gold star at (0,0,0)  (our Solar System)
  * grey   = all HZ-relevant stars in the analysis (a spherical cloud in pc)
  * red    = stars around which a habitable-zone Earth analog is detectable
             (per-star p_det > --threshold) at the current aperture D.

The animation sweeps aperture from small to large; the red cloud grows as D
increases. The view slowly rotates so the 3D structure reads.

Two styles (like the 2D version):
  * paper : white background, black ink       -> reef_cloud3d_paper.gif/.mp4
  * dark  : transparent background, white ink -> reef_cloud3d_dark.gif + black .mp4

Everything is a CLI knob (no hardcoded science params):

    python3 animate_cloud3d.py                       # defaults
    python3 animate_cloud3d.py --cap_pc 100          # zoom to 100 pc
    python3 animate_cloud3d.py --max_stars 8000 --n_frames 48
    python3 animate_cloud3d.py --threshold 0.5 --spin 3 --style dark
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from config import RunConfig
from animate_starmap import (compute_masks, save_opaque_gif,
                             save_transparent_gif, save_mp4)


def radec_to_xyz(ra_deg, dec_deg, dist_pc):
    """Equatorial spherical -> Cartesian [pc], Sun at origin."""
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    x = dist_pc * np.cos(dec) * np.cos(ra)
    y = dist_pc * np.cos(dec) * np.sin(ra)
    z = dist_pc * np.sin(dec)
    return x, y, z


def _style_axis(ax, ink, cap, dark):
    ax.set_xlim(-cap, cap)
    ax.set_ylim(-cap, cap)
    ax.set_zlim(-cap, cap)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_xlabel("x [pc]", color=ink, fontsize=8, labelpad=-6)
    ax.set_ylabel("y [pc]", color=ink, fontsize=8, labelpad=-6)
    ax.set_zlabel("z [pc]", color=ink, fontsize=8, labelpad=-6)
    ax.tick_params(colors=ink, labelsize=6, pad=-2)
    for pane_axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            pane_axis.set_pane_color((0, 0, 0, 0))          # transparent panes
        except Exception:
            pass
        pane_axis.line.set_color(ink if not dark else (1, 1, 1, 0.5))
    ax.grid(False)
    for lbl in (ax.get_xticklabels() + ax.get_yticklabels()
                + ax.get_zticklabels()):
        lbl.set_color(ink)


def render_frame(xyz, masks_this_frame, scenarios, D, threshold, azim, elev,
                 cap, style, figsize, dpi):
    x, y, z = xyz
    dark = (style == "dark")
    ink = "white" if dark else "black"
    grey = "#c8ccd2" if dark else "#6b7178"
    grey_alpha = 0.25 if dark else 0.45
    red = "#ff453a" if dark else "#d62728"
    sun = "#ffd60a"

    fig = plt.figure(figsize=figsize, dpi=dpi)
    if dark:
        fig.patch.set_alpha(0.0)

    for i, (scen, mask) in enumerate(zip(scenarios, masks_this_frame)):
        ax = fig.add_subplot(1, len(scenarios), i + 1, projection="3d")
        if dark:
            ax.set_facecolor("none")
        # grey cloud (non-detectable shown faint)
        ax.scatter(x[~mask], y[~mask], z[~mask], s=2, c=grey, alpha=grey_alpha,
                   linewidths=0, depthshade=True)
        # red detectable
        n_red = int(mask.sum())
        ax.scatter(x[mask], y[mask], z[mask], s=7, c=red, alpha=0.85,
                   linewidths=0, depthshade=True)
        # Sun
        ax.scatter([0], [0], [0], s=90, c=sun, marker="*",
                   edgecolors=ink, linewidths=0.4, depthshade=False)
        ax.view_init(elev=elev, azim=azim)
        _style_axis(ax, ink, cap, dark)
        pct = 100.0 * n_red / mask.size            # fraction of the local cloud
        ax.set_title(f"contrast floor {scen.contrast_floor:.0e}\n"
                     f"detectable: {pct:5.1f}%", color=ink, fontsize=10,
                     fontfamily="monospace")

    fig.suptitle(f"within {cap:.0f} pc  |  aperture diameter = {D:>5,.0f} m",
                 color=ink, fontsize=13, y=0.97, fontfamily="monospace")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return buf


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="results/animations")
    p.add_argument("--style", choices=["paper", "dark", "both"], default="both")
    p.add_argument("--n_frames", type=int, default=36)
    p.add_argument("--dmin", type=float, default=4.0)
    p.add_argument("--dmax", type=float, default=1000.0)
    p.add_argument("--cap_pc", type=float, default=300.0,
                   help="show stars within this distance [pc]")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--exposure_min", type=float, default=None,
                   help="detection integration time [minutes] (default: config 5 min)")
    p.add_argument("--max_stars", type=int, default=8000,
                   help="cap stars for a clean/fast cloud (random subsample)")
    p.add_argument("--n_draws", type=int, default=600)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--hold", type=int, default=6)
    p.add_argument("--spin", type=float, default=0.0, help="deg azimuth per frame (0 = fixed view)")
    p.add_argument("--elev", type=float, default=18.0)
    p.add_argument("--azim0", type=float, default=30.0)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--catalog", default=None)
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cfg = RunConfig()
    if args.catalog:
        cfg.survey.catalog_path = args.catalog
    # restrict the simulated shell to the display cap: dense cloud + fast compute
    cfg.survey.dist_max_pc = args.cap_pc
    if args.exposure_min is not None:      # detection integration time [minutes]
        cfg.observation.exposure_detect_s = args.exposure_min * 60.0

    apertures = np.geomspace(args.dmin, args.dmax, args.n_frames)
    stars, masks, scenarios = compute_masks(
        cfg, base_dir, apertures, args.threshold, args.max_stars, args.n_draws)

    # distance cap for the display cloud
    dist = stars["distance_pc"].to_numpy()
    keep = dist <= args.cap_pc
    n_scen = len(scenarios)
    masks = [[masks[si][fi][keep] for fi in range(len(apertures))]
             for si in range(n_scen)]
    xyz = radec_to_xyz(stars["ra"].to_numpy()[keep],
                       stars["dec"].to_numpy()[keep], dist[keep])
    print(f">> {keep.sum()} stars within {args.cap_pc:.0f} pc on the cloud")

    figsize = (5.0 * n_scen, 5.4)
    dur_ms = int(1000 / args.fps)

    styles = ["paper", "dark"] if args.style == "both" else [args.style]
    for style in styles:
        print(f">> rendering style: {style}")
        frames = []
        for fi, D in enumerate(apertures):
            azim = args.azim0 + args.spin * fi
            frame = render_frame(xyz, [masks[si][fi] for si in range(n_scen)],
                                 scenarios, D, args.threshold, azim, args.elev,
                                 args.cap_pc, style, figsize, args.dpi)
            frames.append(frame)
            print(f"   {fi+1}/{len(apertures)}", end="\r")
        print()
        frames += [frames[-1]] * args.hold

        base = "reef_cloud3d_paper" if style == "paper" else "reef_cloud3d_dark"
        bg = (255, 255, 255) if style == "paper" else (0, 0, 0)
        gif = os.path.join(out_dir, base + ".gif")
        mp4 = os.path.join(out_dir, base + ".mp4")
        # Encode on local disk, then bulk-copy to the (possibly network) out_dir:
        # PIL's incremental writes time out over a Google Drive mount.
        if style == "paper":
            _emit(lambda p: save_opaque_gif(frames, p, dur_ms, bg=bg), gif)
        else:
            _emit(lambda p: save_transparent_gif(frames, p, dur_ms), gif)
        try:
            _emit(lambda p: save_mp4(frames, p, args.fps, bg=bg), mp4)
        except Exception as e:
            print(f"   (mp4 skipped: {e})")
        print(f"   wrote {gif}")

    print(f">> done -> {out_dir}")


def _emit(save_fn, final_path):
    """Run save_fn on a fast local temp path, then copy the finished file to
    final_path in a single write (robust against slow network filesystems)."""
    import shutil
    import tempfile
    ext = os.path.splitext(final_path)[1]
    fd, tmp = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    try:
        save_fn(tmp)
        shutil.copy2(tmp, final_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    main()
