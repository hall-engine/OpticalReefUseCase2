#!/usr/bin/env python3
"""
animate_starmap.py
==================
Animated sky map of the yield analysis, sweeping aperture from small to large.

Three panels side by side, one per contrast floor. In each panel:
  * grey  = all HZ-relevant stars in the analysis (the selected sample)
  * red   = stars around which a habitable-zone Earth analog is detectable
            (per-star detection probability p_det > --threshold) at the
            current aperture D.

Two style variants are rendered:
  * paper : white background, black ink        (reef_starmap_paper.gif/.mp4)
  * dark  : transparent background, white ink   (reef_starmap_dark.gif  +
            solid-black .mp4, since video cannot carry an alpha channel)

Everything is a CLI knob (no hardcoded science params):

    python3 animate_starmap.py                     # full sample, defaults
    python3 animate_starmap.py --max_stars 4000    # faster, sparser map
    python3 animate_starmap.py --n_frames 60 --dmin 4 --dmax 1000
    python3 animate_starmap.py --threshold 0.5 --style both

Frames sweep log-spaced apertures dmin..dmax. The heavy Monte Carlo runs once;
detection masks for every aperture are precomputed, then both styles render
from the same masks.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from config import RunConfig
import catalog
import montecarlo
import photometry


# --------------------------------------------------------------------------
# compute detection masks over the aperture sweep
# --------------------------------------------------------------------------
def compute_masks(cfg: RunConfig, base_dir: str, apertures, threshold,
                  max_stars, n_draws):
    cfg.montecarlo.n_draws = n_draws
    rng = np.random.default_rng(cfg.montecarlo.seed)

    raw = catalog.load_catalog(cfg.survey, base_dir=base_dir)
    # optional down-cap purely for a cleaner/faster map (random, unbiased)
    if max_stars:
        cfg.survey.test_max_stars = max_stars
        stars, _ = catalog.select_stars(raw, cfg.survey, cfg.montecarlo,
                                        test_mode=True, rng=rng)
    else:
        stars, _ = catalog.select_stars(raw, cfg.survey, cfg.montecarlo,
                                        test_mode=False, rng=rng)
    n_stars = len(stars)
    print(f">> {n_stars} stars on the map")

    orb = montecarlo.sample_orbits(
        hz_inner_m=stars["hz_inner_m"].to_numpy(),
        hz_outer_m=stars["hz_outer_m"].to_numpy(),
        distance_m=stars["distance_m"].to_numpy(),
        mc=cfg.montecarlo, rng=rng,
    )
    g_mag = stars["phot_g_mean_mag"].to_numpy()
    lum = stars["lum_flame"].to_numpy()

    # observation mode -> effective bandwidth, SNR threshold, exposure
    bw_frac, target_snr, exposure_s, _ = cfg.observation.effective(cfg.instrument)

    scenarios = cfg.contrast_scenarios
    # masks[scenario_index][frame_index] -> boolean array (n_stars,)
    masks = [[None] * len(apertures) for _ in scenarios]
    for fi, D in enumerate(apertures):
        for si, scen in enumerate(scenarios):
            det = photometry.detectability(orb, g_mag, lum, D,
                                           scen.contrast_floor, exposure_s,
                                           cfg.instrument, bandwidth_frac=bw_frac,
                                           target_snr=target_snr)
            p_det = det["detect"].mean(axis=1)
            masks[si][fi] = p_det > threshold
        print(f"   frame {fi+1}/{len(apertures)}  D = {D:7.1f} m", end="\r")
    print()
    return stars, masks, scenarios


# --------------------------------------------------------------------------
# render one frame to an RGBA array
# --------------------------------------------------------------------------
def render_frame(ra, dec, masks_this_frame, scenarios, D, threshold,
                 style, figsize, dpi):
    dark = (style == "dark")
    ink = "white" if dark else "black"
    grey = "#9aa0a6" if dark else "#b8bcc2"
    red = "#ff453a" if dark else "#d62728"

    fig, axes = plt.subplots(1, len(scenarios), figsize=figsize, dpi=dpi,
                             sharex=True, sharey=True)
    if len(scenarios) == 1:
        axes = [axes]

    if dark:
        fig.patch.set_alpha(0.0)

    for ax, scen, mask in zip(axes, scenarios, masks_this_frame):
        if dark:
            ax.set_facecolor("none")
        else:
            ax.set_facecolor("white")

        ax.scatter(ra, dec, s=3, c=grey, alpha=0.45, linewidths=0, marker=".")
        n_red = int(mask.sum())
        ax.scatter(ra[mask], dec[mask], s=9, c=red, alpha=0.9, linewidths=0,
                   marker="o")

        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_yticks([-90, -45, 0, 45, 90])
        ax.set_title(f"{scen.name}\ncontrast floor {scen.contrast_floor:.0e}",
                     color=ink, fontsize=11)
        ax.set_xlabel("Right ascension [deg]", color=ink, fontsize=9)
        ax.text(0.03, 0.05, f"detectable: {n_red}", transform=ax.transAxes,
                color=red, fontsize=10, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_color(ink)
        ax.tick_params(colors=ink, labelsize=8)

    axes[0].set_ylabel("Declination [deg]", color=ink, fontsize=9)
    fig.suptitle(f"Optical Reef  —  aperture D = {D:,.0f} m"
                 f"   (red: p$_{{det}}$ > {threshold:g})",
                 color=ink, fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return buf


# --------------------------------------------------------------------------
# save helpers
# --------------------------------------------------------------------------
def _composite(rgba, bg):
    """Flatten RGBA onto an opaque background colour (0-255 tuple)."""
    a = rgba[..., 3:4].astype(float) / 255.0
    rgb = rgba[..., :3].astype(float)
    out = rgb * a + np.array(bg, float) * (1 - a)
    return out.astype(np.uint8)


def save_opaque_gif(frames, path, dur_ms, bg=(255, 255, 255)):
    imgs = [Image.fromarray(_composite(f, bg)) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=dur_ms, loop=0, optimize=True)


def save_transparent_gif(frames, path, dur_ms, alpha_thresh=80):
    pal_imgs = []
    for f in frames:
        rgb = Image.fromarray(np.ascontiguousarray(f[..., :3]))
        p = rgb.quantize(colors=255, method=Image.MEDIANCUT)
        arr = np.array(p)
        arr[f[..., 3] < alpha_thresh] = 255           # reserve index 255 = clear
        pim = Image.fromarray(arr)
        pim.putpalette(p.getpalette())
        pal_imgs.append(pim)
    pal_imgs[0].save(path, save_all=True, append_images=pal_imgs[1:],
                     duration=dur_ms, loop=0, transparency=255, disposal=2)


def save_mp4(frames, path, fps, bg):
    """Pipe raw RGB frames straight to the ffmpeg binary (no imageio plugin)."""
    import shutil
    import subprocess
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg binary not found on PATH")
    rgb = [_composite(f, bg) for f in frames]
    h, w = rgb[0].shape[:2]
    if h % 2 or w % 2:                                 # ffmpeg needs even dims
        h -= h % 2
        w -= w % 2
        rgb = [r[:h, :w] for r in rgb]
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for r in rgb:
        proc.stdin.write(np.ascontiguousarray(r).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(proc.stderr.read().decode()[:300])


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="results/animations")
    p.add_argument("--style", choices=["paper", "dark", "both"], default="both")
    p.add_argument("--n_frames", type=int, default=40)
    p.add_argument("--dmin", type=float, default=4.0)
    p.add_argument("--dmax", type=float, default=1000.0)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="p_det above which a star is 'detectable' (red)")
    p.add_argument("--max_stars", type=int, default=None,
                   help="cap stars on the map (random subsample); default full")
    p.add_argument("--n_draws", type=int, default=800)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--hold", type=int, default=6,
                   help="extra repeats of the final frame before looping")
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--catalog", default=None)
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cfg = RunConfig()
    if args.catalog:
        cfg.survey.catalog_path = args.catalog

    apertures = np.geomspace(args.dmin, args.dmax, args.n_frames)

    stars, masks, scenarios = compute_masks(
        cfg, base_dir, apertures, args.threshold, args.max_stars, args.n_draws)
    ra = stars["ra"].to_numpy()
    dec = stars["dec"].to_numpy()

    figsize = (5.2 * len(scenarios), 5.4)
    dur_ms = int(1000 / args.fps)

    styles = ["paper", "dark"] if args.style == "both" else [args.style]
    for style in styles:
        print(f">> rendering style: {style}")
        frames = []
        for fi, D in enumerate(apertures):
            frame = render_frame(ra, dec, [masks[si][fi] for si in range(len(scenarios))],
                                 scenarios, D, args.threshold, style, figsize, args.dpi)
            frames.append(frame)
            print(f"   {fi+1}/{len(apertures)}", end="\r")
        print()
        frames += [frames[-1]] * args.hold          # hold last frame before loop

        if style == "paper":
            gif = os.path.join(out_dir, "reef_starmap_paper.gif")
            mp4 = os.path.join(out_dir, "reef_starmap_paper.mp4")
            save_opaque_gif(frames, gif, dur_ms, bg=(255, 255, 255))
            try:
                save_mp4(frames, mp4, args.fps, bg=(255, 255, 255))
            except Exception as e:
                print(f"   (mp4 skipped: {e})")
            print(f"   wrote {gif}")
        else:
            gif = os.path.join(out_dir, "reef_starmap_dark.gif")
            mp4 = os.path.join(out_dir, "reef_starmap_dark.mp4")
            save_transparent_gif(frames, gif, dur_ms)
            try:
                save_mp4(frames, mp4, args.fps, bg=(0, 0, 0))
            except Exception as e:
                print(f"   (mp4 skipped: {e})")
            print(f"   wrote {gif}  (transparent)")

    print(f">> done -> {out_dir}")


if __name__ == "__main__":
    main()
