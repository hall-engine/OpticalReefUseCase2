"""
plots.py
========
Figures for the direct-imaging yield pipeline. All output is written as PNGs;
nothing is shown interactively (safe on headless HPC nodes).

Figures
-------
completeness_vs_aperture   mean HZ completeness vs aperture, per contrast floor
yield_vs_aperture          expected local HZ-analog yield vs aperture (Drake)
accessible_vs_aperture     fraction of sample with p_det > threshold
completeness_heatmap       aperture x scenario grid of completeness
pdet_distribution          spread of per-star p_det (the orbital-MC signature)
sample_overview            distance & luminosity of the selected stars
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]


def _observatory_markers(ax, sw, cfg, yfield):
    """Mark reference observatories on `yfield` vs aperture, interpolated onto
    the reference contrast-scenario curve (their common ~1e-10 design point)."""
    obs = getattr(cfg, "observatories", None)
    if not obs:
        return
    ref = cfg.observatory_ref_scenario
    s = sw[sw["scenario"] == ref].sort_values("aperture_m")
    if len(s) < 2:
        return
    xs = s["aperture_m"].to_numpy()
    ys = s[yfield].to_numpy()
    logx = np.log10(xs)
    for name, D in obs:
        yv = float(np.interp(np.log10(D), logx, ys))
        ax.scatter([D], [yv], marker="D", s=42, facecolor="white",
                   edgecolor="black", linewidths=1.0, zorder=6)
        ax.annotate(f"{name} ({D:g} m)", (D, yv), textcoords="offset points",
                    xytext=(5, 5), fontsize=7.5, color="black", zorder=6)


def make_all(result, cfg, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    sw = result.sweep_summary
    scen_order = [c.name for c in cfg.contrast_scenarios]

    _completeness_vs_aperture(sw, scen_order, cfg, out_dir)
    _yield_vs_aperture(sw, scen_order, cfg, out_dir)
    _accessible_vs_aperture(sw, scen_order, result.meta, out_dir)
    _completeness_heatmap(sw, cfg, scen_order, out_dir)
    _pdet_distribution(result, cfg, out_dir)
    _sample_overview(result.stars, out_dir)


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)


def _completeness_vs_aperture(sw, scen_order, cfg, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    has_ci = "completeness_lo" in sw.columns
    for k, scen in enumerate(scen_order):
        s = sw[sw["scenario"] == scen].sort_values("aperture_m")
        x = s["aperture_m"].to_numpy()
        y = s["completeness"].to_numpy()
        c = _COLORS[k % len(_COLORS)]
        if has_ci:
            yerr = np.vstack([y - s["completeness_lo"].to_numpy(),
                              s["completeness_hi"].to_numpy() - y])
            ax.errorbar(x, y, yerr=yerr, fmt="o-", color=c, label=scen,
                        capsize=3, elinewidth=1)
        else:
            ax.plot(x, y, "o-", color=c, label=scen)
    _observatory_markers(ax, sw, cfg, "completeness")
    ax.set_xscale("log")
    _style(ax, "Aperture diameter D [m]",
           "HZ completeness  ⟨p$_{det}$⟩",
           "Earth-analog completeness vs aperture")
    ax.legend(title="contrast floor")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "completeness_vs_aperture.png"), dpi=180)
    plt.close(fig)


def _yield_vs_aperture(sw, scen_order, cfg, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    has_ci = "E_yield_local_lo" in sw.columns
    for k, scen in enumerate(scen_order):
        s = sw[sw["scenario"] == scen].sort_values("aperture_m")
        x = s["aperture_m"].to_numpy()
        y = s["E_yield_local"].to_numpy()
        c = _COLORS[k % len(_COLORS)]
        if has_ci:
            yerr = np.vstack([y - s["E_yield_local_lo"].to_numpy(),
                              s["E_yield_local_hi"].to_numpy() - y])
            ax.errorbar(x, y, yerr=yerr, fmt="s-", color=c, label=scen,
                        capsize=3, elinewidth=1)
        else:
            ax.plot(x, y, "s-", color=c, label=scen)
    _observatory_markers(ax, sw, cfg, "E_yield_local")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, "Aperture diameter D [m]",
           "Expected detectable HZ analogs (local pop.)",
           "Drake-term yield vs aperture")
    ax.legend(title="contrast floor")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "yield_vs_aperture.png"), dpi=180)
    plt.close(fig)


def _accessible_vs_aperture(sw, scen_order, meta, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for k, scen in enumerate(scen_order):
        s = sw[sw["scenario"] == scen].sort_values("aperture_m")
        ax.plot(s["aperture_m"].to_numpy(), 100.0 * s["frac_accessible"].to_numpy(), "^-",
                color=_COLORS[k % len(_COLORS)], label=scen)
    ax.set_xscale("log")
    thr = meta.get("accessible_threshold", 0.05)
    _style(ax, "Aperture diameter D [m]",
           f"Accessible stars [%]  (p$_{{det}}$ > {thr})",
           "Fraction of sample with an accessible HZ")
    ax.legend(title="contrast floor")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accessible_vs_aperture.png"), dpi=180)
    plt.close(fig)


def _completeness_heatmap(sw, cfg, scen_order, out_dir):
    apertures = sorted(sw["aperture_m"].unique())
    grid = np.zeros((len(scen_order), len(apertures)))
    for i, scen in enumerate(scen_order):
        for j, D in enumerate(apertures):
            row = sw[(sw["scenario"] == scen) & (sw["aperture_m"] == D)]
            grid[i, j] = row["completeness"].iloc[0] if len(row) else np.nan

    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * len(apertures), 3.5))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=max(grid.max(), 1e-6))
    ax.set_xticks(range(len(apertures)))
    ax.set_xticklabels([f"{a:g}" for a in apertures])
    ax.set_yticks(range(len(scen_order)))
    ax.set_yticklabels(scen_order)
    ax.set_xlabel("Aperture diameter D [m]")
    ax.set_title("HZ completeness ⟨p$_{det}$⟩")
    for i in range(len(scen_order)):
        for j in range(len(apertures)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                    color="w" if grid[i, j] < 0.6 * grid.max() else "k", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "completeness_heatmap.png"), dpi=180)
    plt.close(fig)


def _pdet_distribution(result, cfg, out_dir):
    """Per-star p_det spread at the largest aperture — the orbital-MC signature."""
    ps = result.per_star
    D = max(cfg.apertures_m)
    fig, ax = plt.subplots(figsize=(7, 5))
    for k, scen in enumerate([c.name for c in cfg.contrast_scenarios]):
        vals = ps[(ps["aperture_m"] == D) & (ps["scenario"] == scen)]["p_det"].to_numpy()
        ax.hist(vals, bins=np.linspace(0, 1, 21), histtype="step", linewidth=2,
                color=_COLORS[k % len(_COLORS)], label=scen)
    _style(ax, "Per-star detection probability p$_{det}$", "Number of stars",
           f"Orbital-geometry spread of p$_{{det}}$  (D = {D:g} m)")
    ax.legend(title="contrast floor")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pdet_distribution.png"), dpi=180)
    plt.close(fig)


def _sample_overview(stars, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(stars["distance_pc"].to_numpy(), bins=30, color="#4477aa", alpha=0.85)
    _style(axes[0], "Distance [pc]", "Number of stars", "Selected-sample distances")
    sc = axes[1].scatter(stars["teff_gspphot"].to_numpy(), stars["lum_flame"].to_numpy(),
                         c=stars["distance_pc"].to_numpy(), s=10, cmap="plasma")
    axes[1].set_yscale("log")
    axes[1].invert_xaxis()
    _style(axes[1], "T$_{eff}$ [K]", "Luminosity [L$_\\odot$]", "HR view of selection")
    fig.colorbar(sc, ax=axes[1], label="distance [pc]")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sample_overview.png"), dpi=180)
    plt.close(fig)
