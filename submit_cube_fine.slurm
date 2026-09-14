#!/bin/bash
# ============================================================================
# submit_cube_fine.slurm  --  FINER 4-D yield datacube on CARC, WITH per-cell
# sampling error (bootstrap-equivalent standard error over the stellar sample).
#
#   aperture x contrast x integration time x k_OWA
#
# Every run now also accumulates, per cell, the mean and sum-of-squares of the
# per-star detection probability -> the standard error of completeness over the
# stars (= the large-n bootstrap CI), stored as the `completeness_sem` dataset in
# the cube. It is additive over stars, so this SHARDED array job produces it for
# free; combine merges the shards.
#
# Finer axes than the headline cube: more apertures and integration times.
# Cells = 45 x 5 x 12 x 6 = 16,200 (~2.3x the 7,200-cell cube), so run it as an
# array and merge.
#
# Submit:
#     sbatch submit_cube_fine.slurm
#     # after the array finishes, merge the shards (interactive or a small job):
#     python run_cube.py --combine parts_fine/ --out cube_fine.h5
#     # (combine_cube.slurm defaults to parts/ + cube.h5 -- edit it, or just run
#     #  the one-liner above; merging is cheap and needs no internet.)
#
# Then pull cube_fine.h5 down and locally:
#     python3 cube_figures.py cube_fine.h5 --err_sigma 1
#
# NOTE: gaia_pole_sample.csv is .gitignored -- copy/scp it here first (compute
# nodes have no internet).
# ============================================================================
#SBATCH --job-name=reef_cube_fine
#SBATCH --output=logs/cubefine_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --partition=main
#SBATCH --account=dbarnhar_1592
#SBATCH --array=0-23

module purge
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate optical_reef 2>/dev/null || true
export PATH="$HOME/.conda/envs/optical_reef/bin:$PATH"
echo "using python: $(which python)"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p logs parts_fine

NSHARDS=24

python -u run_cube.py \
    --shard "${SLURM_ARRAY_TASK_ID}" --nshards "${NSHARDS}" --tmp parts_fine \
    --n_apertures 45 --dmin 4 --dmax 1000 \
    --contrasts 1e-9 3e-10 1e-10 3e-11 1e-11 \
    --times_min 5 10 15 30 45 60 90 120 240 360 480 720 \
    --kowa 8 16 32 64 128 0 \
    --mode detection \
    --n_draws 2000 \
    --star_batch 20000
