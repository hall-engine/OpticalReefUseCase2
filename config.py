"""
config.py
=========
Central configuration for the Optical Reef direct-imaging yield pipeline.

Everything the science depends on lives here as dataclasses with documented
defaults, so a run is fully described by (InstrumentConfig, MonteCarloConfig,
SurveyConfig, DrakeConfig) plus a list of ContrastScenario and aperture sizes.

All astrophysical constants are SI unless a name says otherwise.
Stellar luminosity `lum_flame` from Gaia is in units of L_sun.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

# --------------------------------------------------------------------------
# Physical constants (SI)
# --------------------------------------------------------------------------
AU_M = 1.495978707e11        # astronomical unit [m]
PC_M = 3.0856775814913673e16  # parsec [m]
R_EARTH_M = 6.371e6          # Earth radius [m]
R_SUN_M = 6.957e8            # solar radius [m]


# --------------------------------------------------------------------------
# Instrument
# --------------------------------------------------------------------------
@dataclass
class InstrumentConfig:
    """Optical / detector parameters.  Defaults trace to the HabEx & LUVOIR
    concept studies (see instrument.txt in the repo root for justifications)."""

    lambda0_m: float = 550e-9    # central wavelength [m]
    bandwidth: float = 0.20      # fractional bandwidth (dimensionless)
    k_iwa: float = 3.0           # IWA = k_iwa * lambda / D
    k_owa: Optional[float] = 32.0  # OWA = k_owa * lambda / D; None disables it.
    #   The outer working angle is the dark-hole outer radius, set by the
    #   deformable-mirror actuator format: k_owa ~= N_actuators_across / 2.
    #   Default 32 (a 64-actuator DM). This is THE key design knob for large
    #   apertures -- the OWA shrinks in angle as D grows, so a fixed DM caps the
    #   useful aperture and creates a yield sweet spot. Raise it for a larger DM;
    #   set None to remove the outer limit entirely.
    eta_inst: float = 0.20       # cumulative optical throughput (excl. coronagraph mask)
    eta_coron: float = 0.30      # coronagraph planet throughput
    sigma_sys: float = 0.10      # systematic speckle fraction (post-subtraction floor)
    target_snr: float = 10.0     # detection SNR threshold
    f0_v: float = 1e11           # V-band photon zero-point [ph/s/m^2/um] (Vega V=0)
    nzodi_level: float = 3.0     # exozodiacal dust level [zodis]
    m_v_zodi: float = 23.0       # local zodi surface brightness [mag/arcsec^2]

    def iwa_rad(self, D_m: float) -> float:
        return self.k_iwa * self.lambda0_m / D_m

    def owa_rad(self, D_m: float) -> Optional[float]:
        if self.k_owa is None:
            return None
        return self.k_owa * self.lambda0_m / D_m


# --------------------------------------------------------------------------
# Contrast scenarios
# --------------------------------------------------------------------------
@dataclass
class ContrastScenario:
    """A raw stellar-residual contrast floor in the dark hole (flat with
    separation by default)."""
    name: str
    contrast_floor: float


DEFAULT_CONTRAST_SCENARIOS = [
    ContrastScenario("current_1e-9", 1e-9),    # ~ best demonstrated lab contrast
    ContrastScenario("nearterm_1e-10", 1e-10),  # HabWorlds-class target
    ContrastScenario("goal_1e-11", 1e-11),     # Optical Reef aspirational goal
]

# Aperture diameters to sweep [m]. Dense grid; includes the reference-observatory
# sizes (4, 6, 8, 15 m) so their markers land exactly on the curve. Reef ~1 km.
DEFAULT_APERTURES_M = [4.0, 6.0, 8.0, 12.0, 15.0, 20.0, 30.0, 50.0, 80.0,
                       120.0, 200.0, 300.0, 500.0, 700.0, 1000.0]


# --------------------------------------------------------------------------
# Monte Carlo over orbital geometry
# --------------------------------------------------------------------------
@dataclass
class MonteCarloConfig:
    """Per-star orbital-element sampling.  Each star gets `n_draws` synthetic
    Earth-analog placements; detection probability is the fraction that pass."""

    n_draws: int = 2000          # orbital draws per star
    seed: int = 12345

    # Habitable-zone edges scaled from the Sun by sqrt(L/L_sun) [AU at 1 L_sun].
    hz_inner_au: float = 0.95    # conservative inner edge (runaway greenhouse)
    hz_outer_au: float = 1.67    # conservative outer edge (maximum greenhouse)

    # Semi-major-axis sampling within the HZ: "uniform" or "log".
    a_sampling: str = "uniform"

    # Eccentricity: Beta(a,b). Kipping (2013) long-period prior.
    ecc_beta_a: float = 0.867
    ecc_beta_b: float = 3.03
    ecc_max: float = 0.8         # truncate to keep planets inside a plausible HZ orbit

    # Planet properties (Earth analog).
    planet_radius_m: float = R_EARTH_M
    geometric_albedo: float = 0.30


# --------------------------------------------------------------------------
# Survey / catalog
# --------------------------------------------------------------------------
@dataclass
class SurveyConfig:
    """Which stars enter the sample, and how the local DR3 fraction maps to the
    full local population."""

    catalog_path: str = "gaia_pole_sample.csv"  # cached DR3 pull (project root)
    dist_min_pc: float = 10.0
    dist_max_pc: float = 300.0

    # DR3 random_index fraction the catalog represents (for volume extrapolation).
    # If a companion .meta.json exists it overrides this.
    rand_fraction: float = 0.03

    # Stellar selection (Sun-like main-sequence, HZ-relevant).
    logg_min: float = 4.0
    teff_min_k: float = 3900.0
    teff_max_k: float = 5500.0
    lum_min_lsun: float = 0.05
    lum_max_lsun: float = 1.0

    # Fast-test subsampling: cap the number of selected stars actually simulated.
    test_max_stars: Optional[int] = 200


# --------------------------------------------------------------------------
# Drake-equation linkage
# --------------------------------------------------------------------------
@dataclass
class DrakeConfig:
    """Occurrence rate converting HZ-relevant stars into expected habitable
    Earth analogs."""

    eta_earth: float = 0.24      # HZ Earth-analog occurrence per Sun-like star
    #                              (Bryson et al. 2021, Kepler eta_Earth ~ 0.37-0.60;
    #                               0.24 is a conservative mid value)


# --------------------------------------------------------------------------
# Observation mode: detection (broadband) vs characterization (spectroscopy)
# --------------------------------------------------------------------------
@dataclass
class ObservationConfig:
    """What counts as a 'success'.

    detection      -- one broadband photometric measurement over the full band;
                      the planet just has to reach `snr_detect` in that band.
    characterization -- spectroscopy: the band is split into resolution elements
                      of width lambda / R, so each element carries only ~1/R of
                      the light, and the planet must reach `snr_char` PER element.
                      This is what an atmospheric / biosignature measurement needs.

    The mode changes exactly two things fed to the SNR model: the effective
    fractional bandwidth and the SNR threshold (and, in practice, the exposure).
    """

    mode: str = "detection"              # "detection" or "characterization"
    snr_detect: float = 10.0             # broadband detection threshold
    snr_char: float = 10.0               # per-element characterization threshold
    spectral_R: float = 70.0             # resolution elements (lambda/dlambda)
    exposure_detect_s: float = 5.0 * 60.0         # 5 min  (fast survey snapshot)
    exposure_char_s: float = 6.0 * 3600.0         # 6 h    (deep spectroscopy)

    def effective(self, inst: "InstrumentConfig"):
        """Return (bandwidth_fraction, target_snr, exposure_s, label) for the
        active mode."""
        if self.mode == "characterization":
            return (1.0 / self.spectral_R, self.snr_char, self.exposure_char_s,
                    f"characterization (R={self.spectral_R:.0f}, "
                    f"SNR>={self.snr_char:g}/elem)")
        return (inst.bandwidth, self.snr_detect, self.exposure_detect_s,
                f"detection (broadband, SNR>={self.snr_detect:g})")


# Reference observatories to mark on the aperture plots: (name, diameter [m]).
# All target ~1e-10 raw contrast, so they are marked on that scenario curve.
DEFAULT_OBSERVATORIES = [
    ("HabEx", 4.0),
    ("HWO", 6.0),
    ("LUVOIR-B", 8.0),
    ("LUVOIR-A", 15.0),
]


# --------------------------------------------------------------------------
# Full run bundle
# --------------------------------------------------------------------------
@dataclass
class RunConfig:
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    montecarlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    survey: SurveyConfig = field(default_factory=SurveyConfig)
    drake: DrakeConfig = field(default_factory=DrakeConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    apertures_m: list = field(default_factory=lambda: list(DEFAULT_APERTURES_M))
    contrast_scenarios: list = field(
        default_factory=lambda: list(DEFAULT_CONTRAST_SCENARIOS)
    )
    observatories: list = field(default_factory=lambda: list(DEFAULT_OBSERVATORIES))
    # contrast scenario on whose curve the reference observatories are marked
    observatory_ref_scenario: str = "nearterm_1e-10"

    def to_dict(self) -> dict:
        d = {
            "instrument": asdict(self.instrument),
            "montecarlo": asdict(self.montecarlo),
            "survey": asdict(self.survey),
            "drake": asdict(self.drake),
            "observation": asdict(self.observation),
            "apertures_m": list(self.apertures_m),
            "contrast_scenarios": [asdict(c) for c in self.contrast_scenarios],
            "observatories": list(self.observatories),
            "observatory_ref_scenario": self.observatory_ref_scenario,
        }
        return d
