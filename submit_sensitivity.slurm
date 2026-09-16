#!/bin/bash
# ============================================================================
# submit_sensitivity.slurm  --  SYSTEMATIC (model-parameter) error tier.
#
# Runs sensitivity.py: (1) one-at-a-time (OAT) sensitivity -> tornado ranking of
# which assumption drives the yield, and (2) a global Monte Carlo over all
# parameters -> the combined systematic error band. Both at one reference design
# point. This is the error that actually dominates (the bootstrap sampling error
# was negligible), and it is what tightens the Drake-equation terms.
#
# The log shows a live progress bar for the Monte Carlo, so you can `tail -f` it
# and leave the job to run.
#
# Submit:  sbatch submit_sensitivity.slurm
# Watch:   tail -f logs/sensitivity_<jobid>.out
#
# Output (results/sensitivity/): oat.csv, mc_samples.csv, summary.json,
# tornado.png, mc_hist.png  -- pull the folder down and plot locally.
#
# NOTE: gaia_pole_sample.csv is .gitignored; copy it here first (as for the cube).
# ============================================================================
#SBATCH --job-name=reef_sens
#SBATCH --output=logs/sensitivity_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=12:00:00
#SBATCH --partition=main
#SBATCH --account=dbarnhar_1592

module purge
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate optical_reef 2>/dev/null || true
export PATH="$HOME/.conda/envs/optical_reef/bin:$PATH"
echo "using python: $(which python)"

# numpy already uses BLAS threads; give it the cores.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p logs

# --- design (this serial version is the SLOW path; prefer the array for the
#     8-diameter sweep). Edit the diameters/contrast to your headline design. ---
APERTURES="10 20 50 100 250 500 750 1000"   # aperture diameters [m]
CONTRAST=1e-10     # coronagraph contrast floor
KOWA=32            # DM format (64x64); 0 disables the OWA
EXP_MIN=360        # detection integration time [min]

# --- Monte Carlo size / fidelity (sampling noise is negligible, so a star
#     subsample is fine; more draws = smoother yields) ---
N_MC=1500
MAX_STARS=30000
N_DRAWS=600

python -u sensitivity.py \
    --apertures ${APERTURES} --contrast ${CONTRAST} --kowa ${KOWA} \
    --exp_min ${EXP_MIN} --mode detection \
    --n_mc ${N_MC} --max_stars ${MAX_STARS} --n_draws ${N_DRAWS} \
    --out_dir results/sensitivity
rc=$?

if [ $rc -eq 0 ]; then
    echo "SENSITIVITY DONE -> results/sensitivity/ (summary.json has the band)"
else
    echo "SENSITIVITY FAILED (rc=${rc})."
fi
exit $rc
