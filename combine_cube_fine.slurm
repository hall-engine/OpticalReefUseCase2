#!/bin/bash
# ============================================================================
# combine_cube_fine.slurm  --  merge the FINE cube shards into cube_fine.h5.
#
# Companion to submit_cube_fine.slurm (which writes parts_fine/part_*.npz).
# The plain combine_cube.slurm defaults to parts/ -> cube.h5; this one targets
# the fine array's parts_fine/ -> cube_fine.h5, which is what cube_figures.py
# and eta_recovery.py read.
#
#     JID=$(sbatch --parsable submit_cube_fine.slurm)
#     sbatch --dependency=afterok:${JID} combine_cube_fine.slurm
# ============================================================================
#SBATCH --job-name=reef_cubefine_combine
#SBATCH --output=logs/cubefine_combine_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=00:30:00
#SBATCH --partition=main
#SBATCH --account=dbarnhar_1592

module purge
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate optical_reef 2>/dev/null || true
export PATH="$HOME/.conda/envs/optical_reef/bin:$PATH"
echo "using python: $(which python)"

python -u run_cube.py --combine parts_fine --out cube_fine.h5
