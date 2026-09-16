#!/bin/bash
# ============================================================================
# submit_sensitivity_array.slurm  --  PARALLEL systematic-error Monte Carlo.
#
# The global parameter Monte Carlo is embarrassingly parallel (every draw is
# independent), so this array shards the N_MC draws across tasks; each task
# evaluates its slice (with its own log progress bar) and writes a partial.
# A dependent combine job then merges them and runs the OAT + outputs.
#
# ~7 h serial  ->  ~20-30 min wall with 20 tasks.
#
# Submit BOTH (array first, then the combine with a dependency on it):
#     JID=$(sbatch --parsable submit_sensitivity_array.slurm)
#     sbatch --dependency=afterok:${JID} combine_sensitivity.slurm
# Watch:   tail -f logs/sensarr_<jobid>_<task>.out   then  logs/senscomb_<jobid>.out
#
# IMPORTANT: the settings block below MUST match combine_sensitivity.slurm
# (same N_MC, SEED, design, MAX_STARS, N_DRAWS) so the draws line up.
#
# NOTE: gaia_pole_sample.csv is .gitignored; copy it here first.
# ============================================================================
#SBATCH --job-name=reef_sensarr
#SBATCH --output=logs/sensarr_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12GB
#SBATCH --time=03:00:00
#SBATCH --partition=main
#SBATCH --account=dbarnhar_1592
#SBATCH --array=0-49

module purge
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate optical_reef 2>/dev/null || true
export PATH="$HOME/.conda/envs/optical_reef/bin:$PATH"
echo "using python: $(which python)"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

mkdir -p logs parts_sens

# ---- settings (KEEP IN SYNC with combine_sensitivity.slurm) ----
# 50 shards x 8 cores each; sweeps 8 diameters in one pass (orbits sampled once
# per draw, photometry at every aperture). Bump N_MC for smoother distributions
# since the cores are cheap here.
NSHARDS=50
N_MC=3000
SEED=2024
APERTURES="10 20 50 100 250 500 750 1000"
CONTRAST=1e-10
KOWA=32
EXP_MIN=360
MAX_STARS=30000
N_DRAWS=600

python -u sensitivity.py \
    --shard "${SLURM_ARRAY_TASK_ID}" --nshards ${NSHARDS} --tmp parts_sens \
    --n_mc ${N_MC} --seed ${SEED} \
    --apertures ${APERTURES} --contrast ${CONTRAST} --kowa ${KOWA} \
    --exp_min ${EXP_MIN} --mode detection \
    --max_stars ${MAX_STARS} --n_draws ${N_DRAWS}
