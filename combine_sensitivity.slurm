#!/bin/bash
# ============================================================================
# combine_sensitivity.slurm  --  merge the parallel MC shards + run OAT/outputs.
#
# Run AFTER submit_sensitivity_array.slurm (via a dependency). Merges
# parts_sens/part_*.npz into the full ordered mc_samples.csv, runs the (cheap)
# one-at-a-time sensitivity, and writes summary.json + tornado.png + mc_hist.png.
#
#     JID=$(sbatch --parsable submit_sensitivity_array.slurm)
#     sbatch --dependency=afterok:${JID} combine_sensitivity.slurm
#
# The settings block MUST match submit_sensitivity_array.slurm.
# ============================================================================
#SBATCH --job-name=reef_senscomb
#SBATCH --output=logs/senscomb_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=01:00:00
#SBATCH --partition=main
#SBATCH --account=dbarnhar_1592

module purge
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate optical_reef 2>/dev/null || true
export PATH="$HOME/.conda/envs/optical_reef/bin:$PATH"
echo "using python: $(which python)"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

mkdir -p logs

# ---- settings (KEEP IN SYNC with submit_sensitivity_array.slurm) ----
N_MC=1500
SEED=2024
APERTURE=15
CONTRAST=1e-10
KOWA=32
EXP_MIN=360
MAX_STARS=30000
N_DRAWS=600

python -u sensitivity.py --combine --tmp parts_sens \
    --n_mc ${N_MC} --seed ${SEED} \
    --aperture ${APERTURE} --contrast ${CONTRAST} --kowa ${KOWA} \
    --exp_min ${EXP_MIN} --mode detection \
    --max_stars ${MAX_STARS} --n_draws ${N_DRAWS} \
    --out_dir results/sensitivity
rc=$?
if [ $rc -eq 0 ]; then
    echo "SENSITIVITY (parallel) DONE -> results/sensitivity/summary.json"
else
    echo "COMBINE FAILED (rc=${rc})."
fi
exit $rc
