"""
photometry.py
=============
Signal-to-noise model for reflected-light direct imaging.

Given the aperture-independent orbital geometry from montecarlo.py, this layer
adds the aperture and the contrast scenario: it computes photon rates, the
speckle and (exo)zodiacal backgrounds, and the per-draw detection SNR, then
returns a boolean detectability mask.

The photon-budget model matches the validated notebook pipeline
(see instrument.txt); the only change is that the planet contrast is now the
phase-dependent value from each Monte Carlo draw rather than a fixed constant.
"""

from __future__ import annotations

import numpy as np

from config import InstrumentConfig
from montecarlo import OrbitSamples

ARCSEC_PER_RAD = 206265.0


def photon_rate_star(g_mag: np.ndarray, D_m: float, inst: InstrumentConfig) -> np.ndarray:
    """Unobstructed stellar photon rate [ph/s] in the band."""
    area = np.pi * (D_m / 2.0) ** 2
    bw_um = (inst.lambda0_m * 1e6) * inst.bandwidth
    return inst.f0_v * 10.0 ** (-0.4 * g_mag) * area * bw_um


def detectability(orb: OrbitSamples, g_mag: np.ndarray, lum_lsun: np.ndarray,
                  D_m: float, contrast_floor: float, t_exp_s: float,
                  inst: InstrumentConfig, bandwidth_frac: float = None,
                  target_snr: float = None) -> dict:
    """Return per-(star, draw) detectability and SNR for one aperture + floor.

    `bandwidth_frac` is the effective fractional bandwidth for the photon budget
    (the broadband value for detection, or 1/R per spectral element for
    characterization). `target_snr` is the matching threshold. Both fall back to
    the instrument's broadband values when not supplied.

    Broadcasting: g_mag / lum_lsun are length-n_stars, reshaped to (n_stars, 1);
    orbital arrays are (n_stars, n_draws).
    """
    if bandwidth_frac is None:
        bandwidth_frac = inst.bandwidth
    if target_snr is None:
        target_snr = inst.target_snr

    g_mag = np.asarray(g_mag)[:, None]
    lum = np.asarray(lum_lsun)[:, None]

    area = np.pi * (D_m / 2.0) ** 2
    bw_um = (inst.lambda0_m * 1e6) * bandwidth_frac
    n_star = inst.f0_v * 10.0 ** (-0.4 * g_mag) * area * bw_um  # (n_stars, 1)

    # --- planet & speckle count rates ---
    c_p = n_star * orb.contrast * inst.eta_inst * inst.eta_coron
    c_sp = n_star * contrast_floor * inst.eta_inst * inst.eta_coron

    # --- (exo)zodiacal background within the PSF core ---
    omega_psf = np.pi * (1.22 * inst.lambda0_m / D_m) ** 2  # [sr]
    f_zodi = inst.f0_v * 10.0 ** (-0.4 * inst.m_v_zodi) * (ARCSEC_PER_RAD ** 2)
    c_zodi_local = f_zodi * area * bw_um * omega_psf * inst.eta_inst
    c_zb = c_zodi_local * (1.0 + lum * inst.nzodi_level)

    # --- SNR ---
    n_sig = c_p * t_exp_s
    var_poisson = (c_p + c_sp + c_zb) * t_exp_s
    var_syst = (inst.sigma_sys * c_sp * t_exp_s) ** 2
    snr = n_sig / np.sqrt(var_poisson + var_syst)

    # --- working-angle gate (IWA and OWA kept separate for limit budgets) ---
    iwa = inst.iwa_rad(D_m)
    owa = inst.owa_rad(D_m)
    iwa_ok = orb.ang_sep_rad > iwa
    owa_ok = (orb.ang_sep_rad < owa) if owa is not None else np.ones_like(iwa_ok)
    within = iwa_ok & owa_ok
    snr_ok = snr >= target_snr

    detect = within & snr_ok
    return {
        "snr": snr,
        "detect": detect,
        "within_wa": within,
        "iwa_ok": iwa_ok,
        "owa_ok": owa_ok,
        "snr_ok": snr_ok,
        "iwa_rad": iwa,
        "owa_rad": owa,
    }
