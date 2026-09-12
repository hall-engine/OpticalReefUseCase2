# reef_yield — Gaia DR3 direct-imaging yield pipeline for Optical Reef

A from-scratch rewrite of the Optical Reef exoplanet direct-imaging use-case
pipeline. It answers: **for a kilometre-class segmented aperture, what fraction
of nearby Sun-like stars would let us actually image an Earth-analog in the
habitable zone — and how does that translate into Drake-equation yield?**

The key advance over the earlier notebook (`../gaia_pole_sample.csv` era) is that
planets are **no longer assumed to sit face-on at maximum elongation**. Each star
gets a Monte Carlo over orbital geometry (inclination, phase, semi-major axis,
eccentricity), so every star yields a *detection probability* rather than a
binary yes/no. That probability distribution is the physical heart of the
result.

---

## Quick start

```bash
# fast local sanity test — subsamples the cached DR3 catalog, ~3 s
python run.py --test

# full local run against the whole cached catalog (all selected stars)
python run.py --out_dir results_full

# fetch a fresh DR3 slice first (needs internet)
python run.py --fetch --rand_fraction 0.03 --out_dir results_full

# physics invariant checks (no data needed)
python test_physics.py
```

Outputs land in `results/` (or `--out_dir`):

| file | contents |
|------|----------|
| `sweep_summary.csv` | one row per (aperture × contrast floor): completeness, accessible-star counts, expected yield |
| `per_star_pdet.csv` | long format: every star × combo detection probability `p_det` |
| `selected_stars.csv` | the HZ-relevant stellar sample + HZ geometry |
| `run_meta.json` | full config + provenance for the run |
| `figures/*.png` | the six figures below |

---

## Pipeline stages

```
catalog.py     Gaia DR3 load/fetch  ->  Sun-like main-sequence HZ selection
montecarlo.py  per-star orbital Monte Carlo (aperture-independent geometry)
photometry.py  aperture + contrast floor  ->  per-draw SNR & detectability
pipeline.py    roll up to completeness + eta_Earth Drake-term yields
plots.py       figures
run.py         CLI
```

Because the orbital geometry (projected separation, phase, orbital radius) does
**not** depend on the telescope, `montecarlo.sample_orbits` runs **once**; the
aperture × contrast sweep is layered on top cheaply. This is what makes the full
5-aperture × 3-contrast sweep run in seconds.

## The Monte Carlo (montecarlo.py)

Per star, `n_draws` synthetic Earth analogs are drawn with:

| element | distribution | note |
|---------|--------------|------|
| semi-major axis `a` | uniform across the star's HZ | `hz_inner..hz_outer`, scaled by √(L/L☉) |
| eccentricity `e` | Beta(0.867, 3.03), truncated at 0.8 | Kipping (2013) prior |
| inclination `i` | isotropic (uniform in cos i) | 0 = face-on, 90° = edge-on |
| mean anomaly `M` | uniform in [0, 2π) | uniform in time |
| argument of periastron `ω` | uniform in [0, 2π) | |

Kepler's equation is solved (vectorised Newton) for the true anomaly. With the
line of sight along +Z and the node longitude irrelevant to both the
projected-separation magnitude and the phase angle, each draw yields:

- projected angular separation `θ = r·√(cos²u + sin²u·cos²i) / d`
- phase angle `α`, with `cos α = −sin u·sin i`
- Lambertian illuminated fraction `Φ(α) = [sin α + (π−α)cos α] / π`
- reflected contrast `C = A_g · Φ(α) · (R_p / r)²`

## Detection & SNR (photometry.py)

Matches the validated notebook photon budget (see `../instrument.txt`), with the
planet contrast now phase-dependent. A draw is **detected** when it is inside the
working angles *and* clears the SNR threshold:

```
θ > IWA (= k_iwa·λ/D)   [and θ < OWA if enabled]
SNR = C_p·t / √[(C_p + C_speckle + C_zodi)·t + (σ_sys·C_speckle·t)²] ≥ target_SNR
```

`p_det(star)` = fraction of that star's draws that are detected.

## Drake-equation linkage (pipeline.py)

Per (aperture × contrast floor):

- **completeness** = ⟨p_det⟩ over the sample — the probability an Earth-analog in
  the HZ is imaged, averaged over orbital geometry.
- **E_yield_sample** = `η_Earth · Σ p_det` — expected detected HZ analogs in the
  simulated sample (η_Earth is the HZ occurrence rate, default 0.24).
- **E_yield_local** = `E_yield_sample / rand_fraction` — extrapolated to the full
  local population the DR3 random slice represents.

### Confidence intervals (`--bootstrap N`)

There are two independent noise sources: the per-star orbital draws (averaged
into `p_det`, smoothed by raising `--n_draws`) and *which stars* fell in the
sample. `--bootstrap N` quantifies the latter: it resamples the selected-star
list **with replacement** N times and recomputes the aggregates, since the
per-star `p_det` is already fixed by the Monte Carlo (so this is fast — no
re-simulation). It adds `completeness_lo/hi`, `frac_accessible_lo/hi`, and
`E_yield_local_lo/hi` (16–84th percentile by default) to `sweep_summary.csv`,
and draws error bars on the completeness and yield figures. Run it on a full
sample, not `--test`. The point estimate is still the full-sample value; the
bootstrap only measures its uncertainty.

This is the bridge to the Drake terms: completeness constrains what an
Optical-Reef-class survey can *observe* of the underlying `f_p · n_e` population,
and the yield is the expected count of characterisable habitable worlds.

## Figures

- `completeness_vs_aperture.png` — headline: ⟨p_det⟩ vs D, one line per contrast floor
- `yield_vs_aperture.png` — expected local HZ-analog yield vs D (log-log)
- `accessible_vs_aperture.png` — % of sample with p_det above threshold
- `completeness_heatmap.png` — aperture × scenario grid
- `pdet_distribution.png` — the orbital-MC spread of per-star p_det
- `sample_overview.png` — distances and HR view of the selected sample

---

## Configuration

Everything is in `config.py` as dataclasses with documented defaults:
`InstrumentConfig`, `MonteCarloConfig`, `SurveyConfig`, `DrakeConfig`, plus the
aperture list and `ContrastScenario`s. CLI flags override the common ones
(`--n_draws`, `--max_stars`, `--exposure`, `--apertures`, `--eta_earth`,
`--k_owa`). Contrast floors default to **1e-9 / 1e-10 / 1e-11** (current lab /
near-term / Reef goal).

> **Note on OWA.** For very large apertures the *outer* working angle (dark-hole
> radius, set by the deformable mirror) can become the binding constraint rather
> than the IWA. It is **disabled by default** (`k_owa = None`) so results are not
> silently gutted; enable with `--k_owa 32` (or a value matched to the DM format)
> for a realistic large-aperture study.

## Scaling to HPC

The pipeline is already fully vectorised and single-machine friendly. To scale:

1. Fetch a larger DR3 fraction: `python run.py --fetch --rand_fraction 0.3`.
2. Drop `--test` so **all** selected stars run; raise `--n_draws` for smoother
   per-star probabilities (memory ≈ `n_stars × n_draws × 8 bytes × ~6 arrays`).
3. For very large samples, shard the catalog by `source_id` across nodes and sum
   the `E_yield_sample` contributions (they are additive); `sweep_summary` rows
   can be re-aggregated from per-shard `Σ p_det` and star counts. A `submit.slurm`
   stub is included as a starting point.

Memory guide: 200 stars × 2000 draws ≈ 3 MB per array; 50k stars × 5000 draws
≈ 2 GB per array — shard beyond that.
