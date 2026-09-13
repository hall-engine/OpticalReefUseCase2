#!/bin/bash
# ============================================================================
# submit_cube_test.slurm  --  QUICK smoke test. Run this FIRST.
#
# Verifies that the cluster env, the catalog, run_cube.py and HDF5 output all
# work, in a few seconds, BEFORE you commit a long job. Produces cube_test.h5.
# Check: the job exits 0, logs/cube_test_*.out has ">> wrote ... cube_test.h5",
# and `h5ls cube_test.h5` shows the datasets + attributes.
# ============================================================================
#SBATCH --job-name=reef_cube_test
#SBATCH --output=logs/cube_test_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:15:00
#SBATCH --partition=main
#SBATCH --account=dbarnhar_1592

module purge
module load conda
# robust conda activation for a non-interactive (batch) shell:
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate optical_reef
echo "using python: $(which python)"

export OMP_NUM_THREADS=1

mkdir -p logs

# --test = 500 stars, 150 draws, coarse aperture grid -> seconds
python -u run_cube.py --test --out cube_test.h5

echo "TEST DONE -> cube_test.h5"
h5ls -v cube_test.h5 2>/dev/null | head -20 || true
