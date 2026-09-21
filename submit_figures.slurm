#!/bin/bash
# ============================================================================
# submit_figures.slurm  --  regenerate EVERY paper figure from the finished
# cube + sensitivity products. No DR3 refetch, no cube/sensitivity rerun.
#
# Depends on cube_fine.h5 (combine_cube_fine.slurm) AND
# results/sensitivity/summary.json (combine_sensitivity.slurm):
#     sbatch --dependency=afterok:${CUBE_COMB}:${SENS_COMB} submit_figures.slurm
# (submit_all.sh wires these dependencies for you.)
#
# Produces, under results/:
#   cube_figures/     cube_3contrast, cube_combined, cube_kowa_sweetspot,
#                     fig_error*, fig_error_by_time  (from cube_fine.h5)
#   eta_recovery/     eta_recovery.png, eta_precision.png       (Fig 8)
#   mission_compare/  mission_compare.png                       (Fig 9)
#   cloud_grids/      static 3D aperture x contrast grids
#   animations/       reef_cloud3d_{paper,dark}.{gif,mp4}
#   limits/           yield_limits.png
#   galaxy_context/   galaxy_context figures (white column + dark wide)
# (sensitivity figures -- tornado / relsigma / mc_hist -- come out of the
#  combine_sensitivity job itself.)
#
# NOTE: the per-star figure scripts read gaia_pole_sample.csv; the mission
# comparison reads gaia_nearby.csv; both must be the RUWE-clean pulls. The
# galaxy figures read gaia_galaxy_context.csv (unaffected by RUWE).
# ============================================================================
#SBATCH --job-name=reef_figures
#SBATCH --output=logs/figures_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24GB
#SBATCH --time=08:00:00
#SBATCH --partition=main
#SBATCH --account=dbarnhar_1592

module purge
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate optical_reef 2>/dev/null || true
export PATH="$HOME/.conda/envs/optical_reef/bin:$PATH"
echo "using python: $(which python)"

export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p logs results

# Run each figure step independently: record failures but keep going so one
# broken figure never blocks the rest.
set +e
FAILED=""
step () {                          # step "label" cmd args...
    local label="$1"; shift
    echo ""; echo "===== ${label} ====="
    "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "!!! ${label} FAILED (rc=${rc})"
        FAILED="${FAILED} ${label}"
    fi
    return 0
}

step "cube figures (3-contrast, error, sweet spot)" \
    python -u cube_figures.py cube_fine.h5 --err_sigma 1

step "eta recovery (Fig 8)" \
    python -u eta_recovery.py --cube cube_fine.h5 \
        --summary results/sensitivity/summary.json

step "mission comparison (Fig 9)" \
    python -u mission_compare.py

step "static 3D cloud grids" \
    python -u cloud_grid.py --style both

step "3D cloud animations (paper + dark)" \
    python -u animate_cloud3d.py --style both --cap_pc 300 \
        --max_stars 6000 --n_draws 800 --n_frames 48 \
        --out_dir results/animations

step "yield-limit decomposition" \
    python -u plot_limits.py --n_apertures 60

step "galaxy context figures" \
    python -u galaxy_context.py --galaxy --style both --layout both

echo ""
echo "============================================================"
if [ -z "${FAILED}" ]; then
    echo " ALL FIGURES DONE -> results/"
else
    echo " DONE with FAILURES in:${FAILED}"
    echo " (check the log above; other figures still completed.)"
fi
echo "============================================================"
