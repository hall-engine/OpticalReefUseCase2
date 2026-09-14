#!/bin/bash
# ============================================================================
# submit_bootstrap.slurm  --  random-error (bootstrap) estimate for the yield.
#
# This is NOT the full 4-D cube. It runs the 1-D aperture sweep (run.py) ONCE on
# the full 10% Gaia sample and bootstrap-resamples the stars to get confidence
# intervals on completeness and yield. The bootstrap itself is cheap (it just
# resamples the already-computed per-star p_det); the cost is the single MC pass
# over ~80k stars x apertures x contrasts -- ~1/160th of the full cube, so it
# finishes in minutes, not hours.
#
# The sampling (random) error is a smooth, slowly-varying quantity, so evaluating
# it on this representative slice (one integration time, one k_OWA, the 3 contrast
# floors) characterises the band for the whole cube. Keep the 10% sample: a 3%
# run would give a ~1.8x wider CI (fewer stars), i.e. the honest error for a 3%
# survey, not for your 10% result.
#
# Output (results/bootstrap/):
#   sweep_summary.csv    completeness/yield with *_lo/_hi/_mean CI columns
#   figures/             completeness_vs_aperture & yield_vs_aperture WITH errorbars
#
# Submit:  sbatch submit_bootstrap.slurm
# Watch:   tail -f logs/bootstrap_<jobid>.out
# ============================================================================
#SBATCH --job-name=reef_boot
#SBATCH --output=logs/bootstrap_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=04:00:00
#SBATCH --partition=main
#SBATCH --account=dbarnhar_1592

module purge
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate optical_reef 2>/dev/null || true
export PATH="$HOME/.conda/envs/optical_reef/bin:$PATH"
echo "using python: $(which python)"

mkdir -p logs

# --- settings (match these to the cube cell your headline figure highlights) ---
NBOOT=1000        # bootstrap resamples of the stellar sample
NDRAWS=2000       # orbital draws per star (match the cube: 2000)
KOWA=32           # OWA coefficient (cube default; 0 disables the OWA)
EXP_MIN=60        # detection integration time [min] (a representative cube time)

python -u run.py \
    --bootstrap ${NBOOT} \
    --n_draws ${NDRAWS} \
    --k_owa ${KOWA} \
    --exp_detect_min ${EXP_MIN} \
    --mode detection \
    --out_dir results/bootstrap
rc=$?

if [ $rc -eq 0 ]; then
    echo "BOOTSTRAP DONE -> results/bootstrap/sweep_summary.csv (+ figures/)"
    echo "  the *_lo/_hi columns are the ${NBOOT}-resample CI; pull the folder down."
else
    echo "BOOTSTRAP FAILED (rc=${rc})."
fi
exit $rc
