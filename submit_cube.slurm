#!/bin/bash
# ============================================================================
# submit_cube.slurm  --  4-D yield datacube on CARC, SINGLE NODE (no array).
#
#   aperture x contrast x integration time x k_OWA   ->   cube.h5
#
# You do NOT need to parallelise this. The pipeline is vectorised and star-chunked,
# so one node handles it comfortably (measured ~2.3e-5 s per star-draw on the full
# 7200-cell grid):
#     cached 3% sample  (24k stars, 2000 draws)  ~ 20 min
#     full DR3          (~800k stars, 2000 draws) ~ 10 h   (still < 24 h)
# Use submit_cube_array.slurm ONLY if you want faster turnaround or a much bigger
# draw count than one node can finish in the wall-time.
#
# ALWAYS run submit_cube_test.slurm first to confirm env/catalog/output.
#
# NOTE: gaia_pole_sample.csv is .gitignored (not in the repo). Copy it here (scp)
# or, for the full local population, fetch on the LOGIN node first (compute nodes
# have no internet):   python run.py --fetch --rand_fraction 1.0
# ============================================================================
#SBATCH --job-name=reef_cube
#SBATCH --output=logs/cube_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24GB
#SBATCH --time=18:00:00
#SBATCH --partition=main
#SBATCH --account=dbarnhar_1592

module purge
module load conda
source activate optical_reef

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p logs

# --- cube axes / model (all CLI knobs; edit to taste) ------------------------
python -u run_cube.py \
    --out cube.h5 \
    --n_apertures 30 --dmin 4 --dmax 1000 \
    --contrasts 1e-9 3e-10 1e-10 3e-11 1e-11 \
    --times_min 5 15 30 60 120 240 480 720 \
    --kowa 8 16 32 64 128 0 \
    --mode detection \
    --n_draws 2000 \
    --star_batch 20000

echo "DONE -> cube.h5  (download it; build figures locally with cube_figures.py)"
