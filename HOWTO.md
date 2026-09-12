# How to run

Runs offline against the cached Gaia catalog in the project root. No internet needed.

## Everything at once (recommended)

    bash run_all.sh          # local: subsampled, fast (~a few minutes)
    bash run_all.sh full     # whole catalogue, publication settings (slow, HPC)

Produces the complete analysis: detection + characterization sweeps (with
bootstrap CIs and figures), the yield-limit decomposition, and the 3D cloud
animations. `local` and `full` differ only in sample/draw/frame counts — the
physics is identical. OWA is on by default (see below).

## Individual pieces

    python3 run.py --test                       # detection sweep (5 min exposure)
    python3 run.py --test --mode characterization  # R=70 spectroscopy (6 h)
    python3 plot_limits.py                       # IWA/OWA/contrast limit budget
    python3 animate_cloud3d.py                   # 3D local-neighbourhood animation
    python3 test_physics.py                      # physics invariant checks

## Output (under results/)

- **detection/**, **characterization/** — `sweep_summary.csv`, `per_star_pdet.csv`,
  `selected_stars.csv`, `run_meta.json`, and `figures/` (six PNGs each)
- **limits/yield_limits.png** — which constraint binds at each aperture
- **animations/** — `reef_cloud3d_{paper,dark}.{gif,mp4}`

## Key knobs (all CLI, nothing hardcoded)

    --mode detection|characterization      observation type
    --exp_detect_min / --exp_char_min      integration per target [minutes]
    --spectral_R                           resolution for characterization [70]
    --k_owa                                OWA = k_owa·λ/D  (DM format; default 32)
    --apertures / --eta_earth / --n_draws / --max_stars / --bootstrap

**OWA is on by default** (`k_owa = 32` in config.py, a 64-actuator DM). It is the
key large-aperture design knob: the dark-hole outer radius shrinks in angle as D
grows, so a fixed DM caps the useful aperture and creates a yield sweet spot.
Raise `--k_owa` for a larger DM, or `--k_owa 0` to disable the outer limit.

## HPC: the 4-D datacube (CARC)

Build the full trade cube (aperture × contrast × integration time × k_OWA) on the
cluster, download one file, make every figure locally without rerunning.

    # on CARC:
    sbatch submit_cube_test.slurm                  # 1) ALWAYS test first (~seconds)
    #   (optional) bigger sample, on the LOGIN node (compute has no internet):
    python run.py --fetch --rand_fraction 1.0
    sbatch submit_cube.slurm                       # 2) single node -> cube.h5

    # Single node is enough: ~20 min (cached 3% sample) to ~10 h (full DR3), both
    # under the 24 h main limit. Only if you need it faster, use the parallel path:
    #     sbatch submit_cube_array.slurm            # array over star shards
    #     sbatch --dependency=afterok:<JOBID> combine_cube.slurm

    # locally, after downloading cube.h5:
    python3 cube_figures.py cube.h5                # 3-contrast, combined, k_OWA sweet-spot
    python3 cube_figures.py cube.h5 --metric yield --kowa 64 --contrast 1e-11

The cube is a self-describing HDF5 file: ACTUAL completeness + the IWA / OWA /
contrast fractions as datasets, and the full run metadata (all model parameters,
seed, git commit, command, timestamp, config JSON) as file attributes — so a
downloaded cube.h5 is fully reproducible on its own. `gaia_pole_sample.csv` is
gitignored — copy or fetch it on the cluster before running. (Reading cube.h5
locally needs h5py: `pip install h5py`; npz output is available via --format npz.)

Full runs and Gaia fetch: see README.md.
