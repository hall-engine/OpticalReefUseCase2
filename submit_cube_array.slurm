#!/bin/bash
# ============================================================================
# submit_cube_array.slurm  --  4-D yield datacube on CARC (OPTIONAL parallel path)
#
#   aperture x contrast x integration time x k_OWA
#
# You only need this if a single node (submit_cube.slurm) is too slow for your
# sample/draw count. Each array task computes detection COUNTS for its shard of
# the star list and writes parts/part_<id>.npz. Counts are additive, so after the
# array finishes you merge them into the cube with combine_cube.slurm.
#
# Submit:
#     sbatch submit_cube_array.slurm
# Then, after it completes (or with a dependency):
#     sbatch --dependency=afterok:<JOBID> combine_cube.slurm
#     # or interactively:  python run_cube.py --combine parts/ --out cube.h5
#
# NOTE: gaia_pole_sample.csv is .gitignored, so it is NOT in the repo. Before
# running, either copy it to this directory (scp) or fetch a larger DR3 slice on
# the LOGIN node (compute nodes have no internet):
#     python run.py --fetch --rand_fraction 1.0   # full local population
# ============================================================================
#SBATCH --job-name=reef_cube
#SBATCH --output=logs/cube_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --partition=main
#SBATCH --account=dbarnhar_1592
#SBATCH --array=0-15

module purge
module load conda
# robust conda activation for a non-interactive (batch) shell:
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate optical_reef
echo "using python: $(which python)"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p logs parts

NSHARDS=16

# --- cube axes / model (edit to taste; everything is a CLI knob) -------------
python -u run_cube.py \
    --shard "${SLURM_ARRAY_TASK_ID}" --nshards "${NSHARDS}" --tmp parts \
    --n_apertures 30 --dmin 4 --dmax 1000 \
    --contrasts 1e-9 3e-10 1e-10 3e-11 1e-11 \
    --times_min 5 15 30 60 120 240 480 720 \
    --kowa 8 16 32 64 128 0 \
    --mode detection \
    --n_draws 2000 \
    --star_batch 20000
