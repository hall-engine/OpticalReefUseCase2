#!/usr/bin/env python3
"""
cube_figures.py
===============
Build figures from a datacube produced by run_cube.py -- NO Monte Carlo rerun.
Download cube.npz from the cluster and point this at it.

    python cube_figures.py cube.npz                       # default figures
    python cube_figures.py cube.npz --contrast 1e-10 --kowa 32
    python cube_figures.py cube.npz --metric yield

Produces (into --out_dir, default next to the cube):
    cube_3contrast.png     yield vs aperture, 3 contrast panels (Option B)
    cube_combined.png      single-panel limit view for one contrast
    cube_kowa_sweetspot.png  ACTUAL vs aperture per k_OWA + peak-aperture curve
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D

C_IWA, C_OWA, C_SNR = "#1f77b4", "#F5821F", "#d62728"   # OWA = Reef orange


def fmt_time(minutes):
    if minutes < 60:
        return f"{minutes:g} min"
    h = minutes / 60.0
    return f"{h:g} h" if h < 24 else f"{h / 24:g} d"


def _nearest(arr, val):
    return int(np.argmin(np.abs(np.asarray(arr) - val)))


class Cube:
    def __init__(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".h5", ".hdf5"):
            self._from_h5(path)
        else:
            self._from_npz(path)

    def _from_npz(self, path):
        d = np.load(path, allow_pickle=True)
        self.ap = np.asarray(d["apertures_m"], float)
        self.con = np.asarray(d["contrasts"], float)
        self.tmin = np.asarray(d["times_min"], float)
        self.kowa = np.asarray(d["kowa"], float)
        self.comp = np.asarray(d["completeness"], float)
        self.f_iwa = np.asarray(d["frac_iwa"], float)
        self.f_owa = np.asarray(d["frac_owa"], float)
        self.f_snr = np.asarray(d["frac_snr"], float)
        self.n_full = int(d["n_full"])
        self.frac = float(d["rand_fraction"])
        self.eta = float(d["eta_earth"])
        self.cap = float(d["dist_max_pc"])
        self.mode = str(d["mode"])

    def _from_h5(self, path):
        import h5py
        with h5py.File(path, "r") as f:
            self.ap = f["axes/apertures_m"][:].astype(float)
            self.con = f["axes/contrasts"][:].astype(float)
            self.tmin = f["axes/times_min"][:].astype(float)
            self.kowa = f["axes/kowa"][:].astype(float)
            self.comp = f["completeness"][:].astype(float)
            self.f_iwa = f["frac_iwa"][:].astype(float)
            self.f_owa = f["frac_owa"][:].astype(float)
            self.f_snr = f["frac_snr"][:].astype(float)
            self.n_full = int(f.attrs["n_full"])
            self.frac = float(f.attrs["rand_fraction"])
            self.eta = float(f.attrs["eta_earth"])
            self.cap = float(f.attrs["dist_max_pc"])
            self.mode = str(f.attrs["mode"])

    def scale(self, metric):
        """value multiplier + y-axis label for a completeness-fraction array."""
        if metric == "yield":
            return self.eta * self.n_full / self.frac, \
                f"expected yield (local pop., < {self.cap:.0f} pc)"
        if metric == "completeness":
            return 1.0, "HZ completeness"
        return 100.0, f"yield [% of viable candidates within {self.cap:.0f} pc]"


def _shade(tmin):
    tcmap = LinearSegmentedColormap.from_list(
        "act", plt.cm.Greys(np.linspace(0.40, 1.0, 256)))
    lo, hi = float(min(tmin)), float(max(tmin))
    norm = LogNorm(vmin=lo, vmax=hi) if hi > lo else None
    return tcmap, norm


def _cbar(fig, ax, tcmap, norm, tmin):
    if norm is None:
        return
    sm = ScalarMappable(norm=norm, cmap=tcmap)
    sm.set_array(np.asarray(tmin, float))
    cb = fig.colorbar(sm, ax=ax, pad=0.012, aspect=28)
    cb.set_label("integration time")
    cb.set_ticks(list(tmin))
    cb.set_ticklabels([fmt_time(t) for t in tmin])


def _ylims(ax, metric, ymax):
    if metric == "percent":
        ax.set_ylim(0, 105)
    else:
        ax.set_yscale("log")
        ax.set_ylim(1.0, ymax * 1.4)


def fig_3contrast(cube, args, out_dir):
    ki = _nearest(cube.kowa, args.kowa)
    mul, ylab = cube.scale(args.metric)
    tcmap, norm = _shade(cube.tmin)
    ymax = max(cube.f_iwa.max(), cube.f_owa[:, ki].max()) * mul

    # which contrast floors to show as panels (nearest in the cube), de-duped
    cidx = []
    for c in args.panels:
        j = _nearest(cube.con, c)
        if j not in cidx:
            cidx.append(j)

    fig, axes = plt.subplots(1, len(cidx), figsize=(5.6 * len(cidx), 5.4),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ci in zip(axes, cidx):
        ax.plot(cube.ap, cube.f_iwa * mul, "--", color=C_IWA, lw=1.8)
        if cube.kowa[ki] > 0:
            ax.plot(cube.ap, cube.f_owa[:, ki] * mul, "--", color=C_OWA, lw=1.8)
        for ei, t in enumerate(cube.tmin):
            col = tcmap(1.0) if norm is None else tcmap(norm(t))
            ax.plot(cube.ap, cube.comp[:, ci, ei, ki] * mul, "-", color=col, lw=2.4)
        ax.set_xscale("log")
        _ylims(ax, args.metric, ymax)
        ax.set_xlabel("Aperture diameter D [m]")
        ax.set_title(f"contrast floor {cube.con[ci]:.0e}", fontsize=11)
        ax.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel(ylab)
    handles = [Line2D([0], [0], color=C_IWA, lw=1.8, ls="--", label="IWA"),
               Line2D([0], [0], color="black", lw=2.4, label="actual yield")]
    if cube.kowa[ki] > 0:
        handles.insert(1, Line2D([0], [0], color=C_OWA, lw=1.8, ls="--",
                                 label=f"OWA (k={cube.kowa[ki]:g})"))
    axes[0].legend(handles=handles, loc="upper left", fontsize=8.5)
    _cbar(fig, list(axes), tcmap, norm, cube.tmin)
    fig.suptitle("yield vs aperture across contrast floors", fontsize=13)
    out = os.path.join(out_dir, "cube_3contrast.png")
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f">> wrote {out}")


def fig_combined(cube, args, out_dir):
    ci = _nearest(cube.con, args.contrast)
    ki = _nearest(cube.kowa, args.kowa)
    mul, ylab = cube.scale(args.metric)
    tcmap, norm = _shade(cube.tmin)
    ymax = max(cube.f_iwa.max(), cube.f_owa[:, ki].max()) * mul

    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.plot(cube.ap, cube.f_iwa * mul, "--", color=C_IWA, lw=2.0)
    if cube.kowa[ki] > 0:
        ax.plot(cube.ap, cube.f_owa[:, ki] * mul, "--", color=C_OWA, lw=2.0)
    for ei, t in enumerate(cube.tmin):
        a = 0.30 + 0.70 * (np.log10(t) - np.log10(cube.tmin.min())) / \
            max(np.log10(cube.tmin.max()) - np.log10(cube.tmin.min()), 1e-9)
        ax.plot(cube.ap, cube.f_snr[:, ci, ei] * mul, "-", color=C_SNR, lw=1.5, alpha=a)
        col = tcmap(1.0) if norm is None else tcmap(norm(t))
        ax.plot(cube.ap, cube.comp[:, ci, ei, ki] * mul, "-", color=col, lw=2.6)
    ax.set_xscale("log")
    _ylims(ax, args.metric, ymax)
    ax.set_xlabel("Aperture diameter D [m]")
    ax.set_ylabel(ylab)
    ax.set_title(f"what limits the yield  —  contrast {cube.con[ci]:.0e}", fontsize=12)
    ax.grid(True, which="both", alpha=0.2)
    handles = [Line2D([0], [0], color=C_IWA, lw=2, ls="--", label="IWA (resolution)"),
               Line2D([0], [0], color=C_SNR, lw=2, label="contrast / SNR"),
               Line2D([0], [0], color="black", lw=2.6, label="actual yield")]
    if cube.kowa[ki] > 0:
        handles.insert(1, Line2D([0], [0], color=C_OWA, lw=2, ls="--",
                                 label=f"OWA (k={cube.kowa[ki]:g})"))
    ax.legend(handles=handles, loc="upper left", fontsize=9, title="limit")
    _cbar(fig, ax, tcmap, norm, cube.tmin)
    fig.tight_layout()
    out = os.path.join(out_dir, "cube_combined.png")
    fig.savefig(out, dpi=190)
    plt.close(fig)
    print(f">> wrote {out}")


def fig_kowa_sweetspot(cube, args, out_dir):
    """The design result: how the sweet-spot aperture moves with k_OWA."""
    ci = _nearest(cube.con, args.contrast)
    ei = _nearest(cube.tmin, args.time_min)
    mul, ylab = cube.scale(args.metric)
    kmask = cube.kowa > 0
    kvals = cube.kowa[kmask]
    kidx = np.where(kmask)[0]
    cmap = plt.cm.viridis
    knorm = LogNorm(vmin=kvals.min(), vmax=kvals.max()) if len(kvals) > 1 else None

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
    peak_ap = []
    for k, ki in zip(kvals, kidx):
        y = cube.comp[:, ci, ei, ki] * mul
        col = cmap(knorm(k)) if knorm else cmap(0.5)
        axL.plot(cube.ap, y, "-", color=col, lw=2.2)
        peak_ap.append(cube.ap[int(np.argmax(y))])
    axL.set_xscale("log")
    _ylims(axL, args.metric, cube.f_iwa.max() * mul)
    axL.set_xlabel("Aperture diameter D [m]")
    axL.set_ylabel(ylab)
    axL.set_title(f"actual yield vs aperture per k$_{{OWA}}$\n"
                  f"(contrast {cube.con[ci]:.0e}, {fmt_time(cube.tmin[ei])})",
                  fontsize=11)
    axL.grid(True, which="both", alpha=0.2)
    if knorm:
        sm = ScalarMappable(norm=knorm, cmap=cmap); sm.set_array(kvals)
        cb = fig.colorbar(sm, ax=axL, pad=0.012); cb.set_label("k$_{OWA}$ (= N$_{act}$/2)")

    axR.plot(kvals, peak_ap, "o-", color="black", lw=2)
    axR.set_xscale("log"); axR.set_yscale("log")
    axR.set_xlabel("k$_{OWA}$  (= N$_{actuators}$ / 2)")
    axR.set_ylabel("sweet-spot aperture [m]")
    axR.set_title("optimal aperture vs DM actuator format", fontsize=11)
    axR.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    out = os.path.join(out_dir, "cube_kowa_sweetspot.png")
    fig.savefig(out, dpi=190)
    plt.close(fig)
    print(f">> wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cube")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--metric", choices=["percent", "yield", "completeness"],
                   default="percent")
    p.add_argument("--contrast", type=float, default=1e-10)
    p.add_argument("--panels", type=float, nargs="+", default=[1e-9, 1e-10, 1e-11],
                   help="contrast floors to show as panels in the 3-contrast figure")
    p.add_argument("--kowa", type=float, default=32.0)
    p.add_argument("--time_min", type=float, default=360.0)
    args = p.parse_args()

    cube = Cube(args.cube)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.cube)) or "."
    os.makedirs(out_dir, exist_ok=True)
    print(f">> cube {cube.comp.shape}  ap={len(cube.ap)} con={len(cube.con)} "
          f"time={len(cube.tmin)} kowa={len(cube.kowa)}  n_full={cube.n_full}")
    fig_3contrast(cube, args, out_dir)
    fig_combined(cube, args, out_dir)
    fig_kowa_sweetspot(cube, args, out_dir)


if __name__ == "__main__":
    main()
