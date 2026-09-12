"""
test_physics.py
===============
Fast invariant checks on the orbital Monte Carlo. Run: python test_physics.py
No network or catalog needed.
"""

import numpy as np

import montecarlo as mc_mod
from config import MonteCarloConfig, AU_M, PC_M, R_EARTH_M


def _ok(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def test_kepler():
    e = np.linspace(0, 0.8, 50)
    M = np.linspace(0, 2 * np.pi, 50)
    E = mc_mod._solve_kepler(M, e)
    resid = np.abs((E - e * np.sin(E)) - np.mod(M, 2 * np.pi))
    _ok("Kepler residual < 1e-10", resid.max() < 1e-10)


def test_lambertian_phase():
    # Phi(0)=1, Phi(pi)=0, Phi(pi/2)=1/pi
    def phi(a):
        return (np.sin(a) + (np.pi - a) * np.cos(a)) / np.pi
    _ok("Lambertian Phi(0)=1", abs(phi(0.0) - 1.0) < 1e-12)
    _ok("Lambertian Phi(pi)=0", abs(phi(np.pi)) < 1e-12)
    _ok("Lambertian Phi(pi/2)=1/pi", abs(phi(np.pi / 2) - 1 / np.pi) < 1e-12)


def test_projection_bounds():
    # Circular orbit, random isotropic inclination: projected sep is bounded
    # above by a/d (face-on) and spreads below it as inclination increases.
    mc = MonteCarloConfig(n_draws=5000, ecc_max=0.0)
    a_au = 1.0
    hz = np.array([a_au * AU_M])          # inner==outer -> a fixed at 1 AU
    dist = np.array([10.0 * PC_M])
    orb = mc_mod.sample_orbits(hz, hz, dist, mc, rng=np.random.default_rng(0))
    sep = orb.ang_sep_rad[0]
    face_on = a_au * AU_M / dist[0]
    _ok("projected sep <= face-on a/d", sep.max() <= face_on * (1 + 1e-9))
    _ok("projected sep spreads below face-on", sep.min() < 0.3 * face_on)
    # Some draws reach near the face-on maximum (near i=0 or near quadrature).
    _ok("some draws near face-on max", sep.max() > 0.95 * face_on)


def test_isotropic_inclination():
    mc = MonteCarloConfig(n_draws=20000)
    hz_in = np.array([0.95 * AU_M])
    hz_out = np.array([1.67 * AU_M])
    dist = np.array([15.0 * PC_M])
    orb = mc_mod.sample_orbits(hz_in, hz_out, dist, mc, rng=np.random.default_rng(1))
    # Mean phase angle over isotropic i and uniform phase should be ~90 deg.
    mean_alpha_deg = np.degrees(orb.phase_angle_rad).mean()
    _ok("mean phase angle ~ 90 deg (isotropic)", abs(mean_alpha_deg - 90.0) < 3.0)
    # Contrast positive and finite.
    _ok("contrast finite & positive", np.all(np.isfinite(orb.contrast)) and np.all(orb.contrast > 0))


def test_contrast_scale():
    # Earth at 1 AU, quadrature: A_g/pi * (R_earth/AU)^2 ~ 1.1e-10 (order check)
    ag = 0.3
    c = ag * (1 / np.pi) * (R_EARTH_M / AU_M) ** 2
    _ok("Earth quadrature contrast ~1e-10", 5e-11 < c < 3e-10)


if __name__ == "__main__":
    test_kepler()
    test_lambertian_phase()
    test_projection_bounds()
    test_isotropic_inclination()
    test_contrast_scale()
    print("\nAll physics invariants passed.")
