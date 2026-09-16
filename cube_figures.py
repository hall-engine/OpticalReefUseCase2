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
        self.sem = (np.asarray(d["completeness_sem"], float)
                    if "completeness_sem" in d.files else None)
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
            self.sem = (f["completeness_sem"][:].astype(float)
                        if "completeness_sem" in f else None)
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
            y = cube.comp[:, ci, ei, ki] * mul
            ax.plot(cube.ap, y, "-", color=col, lw=2.4)
            if cube.sem is not None and args.err_sigma > 0:
                e = cube.sem[:, ci, ei, ki] * mul * args.err_sigma
                ax.fill_between(cube.ap, y - e, y + e, color=col, alpha=0.25,
                                linewidth=0)
        # discrete 1-sigma sampling-error bars on the longest-exposure curve,
        # magnified by --err_inflate so the (tiny) bootstrap error is visible.
        if cube.sem is not None and args.err_bars and args.err_sigma > 0:
            ei = len(cube.tmin) - 1
            y = cube.comp[:, ci, ei, ki] * mul
            e = cube.sem[:, ci, ei, ki] * mul * args.err_sigma * args.err_inflate
            keep = np.where(y > 0.02 * mul)[0]
            pick = keep[np.linspace(0, len(keep) - 1, 8).astype(int)] if len(keep) else []
            ax.errorbar(cube.ap[pick], y[pick], yerr=e[pick], fmt="o", ms=3.5,
                        color="black", ecolor="black", elinewidth=1.1,
                        capsize=3, capthick=1.1, zorder=9)
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
    if cube.sem is not None and args.err_bars:
        # honest label: state the true 1-sigma and any magnification
        med_pp = float(np.median(cube.sem[cube.comp > 0.02])) * 100.0
        note = (f"error bars: {args.err_sigma:g}$\\sigma$ bootstrap sampling error"
                f" (median 1$\\sigma\\approx${med_pp:.2f} pp of candidates)")
        if args.err_inflate != 1.0:
            note += f", shown $\\times${args.err_inflate:g} for visibility"
        fig.text(0.5, 0.005, note, ha="center", fontsize=9, style="italic")
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


def fig_error(cube, args, out_dir):
    """Dedicated sampling-error figure: the bootstrap 1-sigma on the yield as a
    quantity in its own right, vs aperture, one line per contrast floor.
    Left = absolute (percentage points of candidates); right = relative (% of
    the yield value). At the reference integration time and k_OWA."""
    if cube.sem is None:
        print(">> no completeness_sem in cube; skip error figure")
        return
    ki = _nearest(cube.kowa, args.kowa)
    ei = _nearest(cube.tmin, args.time_min)
    z = args.err_sigma
    cidx = []
    for c in args.panels:
        j = _nearest(cube.con, c)
        if j not in cidx:
            cidx.append(j)
    cols = plt.cm.viridis(np.linspace(0.15, 0.85, len(cidx)))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 4.7))
    for col, ci in zip(cols, cidx):
        comp = cube.comp[:, ci, ei, ki]
        sem = cube.sem[:, ci, ei, ki] * z
        lab = f"{cube.con[ci]:.0e}"
        axL.plot(cube.ap, sem * 100.0, "-o", color=col, lw=2, ms=3, label=lab)
        rel = np.where(comp > 0.01, 100.0 * sem / np.maximum(comp, 1e-9), np.nan)
        axR.plot(cube.ap, rel, "-o", color=col, lw=2, ms=3, label=lab)

    for ax in (axL, axR):
        ax.set_xscale("log")
        ax.set_xlabel("Aperture diameter D [m]")
        ax.grid(True, which="both", alpha=0.2)
    axL.set_ylabel(f"{z:g}$\\sigma$ sampling error  [pp of candidates]")
    axL.set_title("absolute bootstrap error", fontsize=11)
    axR.set_ylabel(f"relative {z:g}$\\sigma$ error  [% of yield]")
    axR.set_yscale("log")
    axR.set_title("relative bootstrap error", fontsize=11)
    axL.legend(title="contrast floor", fontsize=8.5, loc="upper right")
    fig.suptitle(f"Bootstrap sampling error on the yield  "
                 f"(t = {fmt_time(cube.tmin[ei])}, k$_{{OWA}}$ = {cube.kowa[ki]:g}, "
                 f"N$_\\star$ = {cube.n_full:,})", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(out_dir, "cube_error.png")
    fig.savefig(out, dpi=190)
    plt.close(fig)
    print(f">> wrote {out}")


def fig_error_by_time(cube, args, out_dir):
    """One COLUMN-WIDTH figure per contrast floor: 1-sigma sampling error vs
    aperture, one line per integration time (black -> orange). The error is a
    +/- on the yield, explained on the side rather than as an axis unit."""
    if cube.sem is None:
        print(">> no completeness_sem in cube; skip per-time error figures")
        return
    ki = _nearest(cube.kowa, args.kowa)
    z = args.err_sigma
    bo = LinearSegmentedColormap.from_list(
        "reef_bo", ["#141414", "#8a2b00", "#d24a00", "#ff9a1f"])
    lo, hi = float(cube.tmin.min()), float(cube.tmin.max())
    norm = LogNorm(vmin=lo, vmax=hi) if hi > lo else None

    cidx = []
    for c in args.panels:
        j = _nearest(cube.con, c)
        if j not in cidx:
            cidx.append(j)

    for ci in cidx:
        fig, ax = plt.subplots(figsize=(3.7, 3.1), dpi=args.dpi)
        for ei, t in enumerate(cube.tmin):
            col = bo(1.0) if norm is None else bo(norm(t))
            ax.plot(cube.ap, cube.sem[:, ci, ei, ki] * z * 100.0, "-",
                    color=col, lw=1.8)
        ax.set_xscale("log")
        ax.set_xlabel("Aperture diameter D [m]", fontsize=11)
        ax.set_ylabel(f"{z:g}$\\sigma$ yield uncertainty  [$\\pm$ pp]", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.set_title(f"Sampling error  —  contrast {cube.con[ci]:.0e}",
                     fontsize=11)
        ax.grid(True, which="both", alpha=0.18)
        if norm is not None:
            sm = ScalarMappable(norm=norm, cmap=bo)
            sm.set_array(cube.tmin)
            cb = fig.colorbar(sm, ax=ax, pad=0.04, aspect=26)
            cb.set_ticks(list(cube.tmin))
            cb.set_ticklabels([f"{int(t)}" for t in cube.tmin])
            cb.ax.tick_params(labelsize=5.5, length=2)
            cb.set_label("exposure [min]", fontsize=7)
        fig.subplots_adjust(left=0.17, right=0.91, top=0.91, bottom=0.16)
        tag = f"{cube.con[ci]:.0e}".replace("-", "m").replace("+", "")
        out = os.path.join(out_dir, f"cube_error_{tag}.png")
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
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
    p.add_argument("--dpi", type=int, default=300,
                   help="output resolution for the column error figures")
    p.add_argument("--err_sigma", type=float, default=1.0,
                   help="shade a +/- N-sigma sampling-error band (bootstrap SEM) "
                        "around each yield curve; 0 disables")
    p.add_argument("--err_bars", action="store_true",
                   help="overlay discrete 1-sigma error-bar markers on the "
                        "longest-exposure curve (3-contrast figure)")
    p.add_argument("--err_inflate", type=float, default=1.0,
                   help="magnify the drawn error bars by this factor for "
                        "visibility (annotated on the figure); true 1-sigma is "
                        "err_inflate=1")
    args = p.parse_args()

    cube = Cube(args.cube)
    # default output location, baked in so figures always land in the same place
    default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "results", "cube_figures")
    out_dir = args.out_dir or default_out
    os.makedirs(out_dir, exist_ok=True)
    print(f">> cube {cube.comp.shape}  ap={len(cube.ap)} con={len(cube.con)} "
          f"time={len(cube.tmin)} kowa={len(cube.kowa)}  n_full={cube.n_full}")
    fig_3contrast(cube, args, out_dir)
    fig_combined(cube, args, out_dir)
    fig_kowa_sweetspot(cube, args, out_dir)
    fig_error(cube, args, out_dir)
    fig_error_by_time(cube, args, out_dir)


if __name__ == "__main__":
    main()
