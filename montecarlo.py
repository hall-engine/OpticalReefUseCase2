"""
montecarlo.py
=============
The core addition over the old face-on pipeline: a proper Monte Carlo over
orbital geometry.

For every selected star we draw `n_draws` synthetic Earth-analog orbits and
compute, for each, the quantities a coronagraph actually cares about:

    * projected angular separation on the sky   [rad]
    * instantaneous star-planet distance          [m]  (drives reflected flux)
    * phase angle alpha (star-planet-observer)     [rad]
    * Lambertian illuminated fraction Phi(alpha)   [-]
    * planet/star flux contrast                    [-]

Orbital elements drawn per (star, draw):
    a   -- semi-major axis, uniform (or log) across the star's HZ
    e   -- eccentricity, truncated Beta(a,b)  (Kipping 2013 default)
    i   -- inclination, isotropic  (uniform in cos i)
    M   -- mean anomaly, uniform in [0, 2pi)  (== uniform in time)
    w   -- argument of periastron, uniform in [0, 2pi)

The longitude of the ascending node is irrelevant: it only rotates the orbit
within the sky plane and changes neither the projected-separation magnitude nor
the phase angle, so it is not sampled.

All outputs are aperture-independent, so this runs ONCE and the aperture x
contrast sweep is layered on cheaply downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import MonteCarloConfig


@dataclass
class OrbitSamples:
    """Per-(star, draw) geometry. Every array has shape (n_stars, n_draws)."""
    ang_sep_rad: np.ndarray     # projected angular separation [rad]
    contrast: np.ndarray        # planet/star flux ratio [-]
    phase_angle_rad: np.ndarray  # star-planet-observer angle [rad]
    sep_m: np.ndarray           # instantaneous star-planet distance [m]

    @property
    def shape(self):
        return self.ang_sep_rad.shape


def _solve_kepler(M: np.ndarray, e: np.ndarray, n_iter: int = 12) -> np.ndarray:
    """Solve E - e sin E = M (vectorised Newton-Raphson)."""
    M = np.mod(M, 2.0 * np.pi)
    E = M + e * np.sin(M)  # good first guess for moderate e
    for _ in range(n_iter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        E = E - f / fp
    return E


def sample_orbits(hz_inner_m: np.ndarray, hz_outer_m: np.ndarray,
                  distance_m: np.ndarray, mc: MonteCarloConfig,
                  rng: np.random.Generator | None = None) -> OrbitSamples:
    """Draw mc.n_draws orbits for each star and return the observable geometry.

    Parameters are 1-D arrays of length n_stars (metres); the return arrays are
    (n_stars, n_draws).
    """
    rng = rng or np.random.default_rng(mc.seed)
    n_stars = hz_inner_m.shape[0]
    n = mc.n_draws
    shape = (n_stars, n)

    # ---- semi-major axis across each star's HZ -------------------------
    a_in = hz_inner_m[:, None]
    a_out = hz_outer_m[:, None]
    u = rng.random(shape)
    if mc.a_sampling == "log":
        a = a_in * (a_out / a_in) ** u
    else:  # uniform in a
        a = a_in + (a_out - a_in) * u

    # ---- eccentricity: truncated Beta ----------------------------------
    e = rng.beta(mc.ecc_beta_a, mc.ecc_beta_b, size=shape)
    e = np.minimum(e, mc.ecc_max)

    # ---- inclination: isotropic (uniform in cos i) ---------------------
    cos_i = rng.uniform(-1.0, 1.0, size=shape)
    sin_i = np.sqrt(np.clip(1.0 - cos_i ** 2, 0.0, 1.0))

    # ---- phase (mean anomaly, uniform in time) & periastron arg --------
    M = rng.uniform(0.0, 2.0 * np.pi, size=shape)
    w = rng.uniform(0.0, 2.0 * np.pi, size=shape)

    # ---- solve for position on the orbit -------------------------------
    E = _solve_kepler(M, e)
    # true anomaly
    nu = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(E / 2.0),
                          np.sqrt(1.0 - e) * np.cos(E / 2.0))
    r = a * (1.0 - e * np.cos(E))          # instantaneous star-planet distance [m]

    # argument of latitude (periastron + true anomaly)
    theta = nu + w
    sin_u = np.sin(theta)
    cos_u = np.cos(theta)

    # Line of sight along +Z; sky plane = (X, Y). Node longitude set to 0.
    #   X = r cos u ; Y = r sin u cos i ; Z = r sin u sin i
    # Projected separation magnitude:
    proj = r * np.sqrt(cos_u ** 2 + (sin_u * cos_i) ** 2)
    ang_sep = proj / distance_m[:, None]

    # Phase angle: cos(alpha) = -Z/r = -sin u sin i  (alpha=0 -> full phase)
    cos_alpha = np.clip(-sin_u * sin_i, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)

    # Lambertian phase function
    phi = (np.sin(alpha) + (np.pi - alpha) * np.cos(alpha)) / np.pi

    # Reflected-light contrast: A_g * Phi(alpha) * (Rp / r)^2
    contrast = mc.geometric_albedo * phi * (mc.planet_radius_m / r) ** 2

    return OrbitSamples(ang_sep_rad=ang_sep, contrast=contrast,
                        phase_angle_rad=alpha, sep_m=r)
