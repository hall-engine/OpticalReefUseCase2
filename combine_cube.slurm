#!/bin/bash
# ============================================================================
# combine_cube.slurm  --  merge the shard counts into the final datacube.
# Run after submit_cube.slurm:
#     sbatch --dependency=afterok:<ARRAY_JOBID> combine_cube.slurm
# Produces cube.h5 (self-describing: all run metadata is stored as HDF5 file
# attributes) + cube.json sidecar. Download cube.h5 to make figures with
# cube_figures.py locally -- no rerun needed.
# ============================================================================
#SBATCH --job-name=reef_cube_combine
#SBATCH --output=logs/cube_combine_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=00:30:00
#SBATCH --partition=main
#SBATCH --account=dbarnhar_1592

module purge
module load conda
# robust conda activation for a non-interactive (batch) shell:
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate optical_reef 2>/dev/null || true
# guarantee the env python even if conda activate misbehaves in batch:
export PATH="$HOME/.conda/envs/optical_reef/bin:$PATH"
echo "using python: $(which python)"

python -u run_cube.py --combine parts --out cube.h5
