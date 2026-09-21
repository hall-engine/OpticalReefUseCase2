#!/bin/bash
# ============================================================================
# submit_all.sh  --  ONE command to reproduce the whole paper on CARC.
#
# Run this on a LOGIN node (it needs internet for the DR3 fetch; compute nodes
# do not). It:
#   0. re-fetches the RUWE-clean catalogs           (login node, here & now)
#        gaia_pole_sample.csv  (10% volume sample, 10-300 pc)   <- fetch_local.py
#        gaia_nearby.csv       (complete <=40 pc census)        <- fetch_nearby.py
#   1. submits the fine 4-D cube array              (submit_cube_fine.slurm)
#   2. combines it -> cube_fine.h5                  (combine_cube_fine.slurm)
#   3. submits the sensitivity MC array             (submit_sensitivity_array.slurm)
#   4. combines it -> results/sensitivity/summary.json (combine_sensitivity.slurm)
#   5. regenerates every figure once 2 & 4 are done (submit_figures.slurm)
# Stages 1-2 and 3-4 run in parallel; the figures wait on both.
#
# Usage:
#     bash submit_all.sh                 # full re-fetch + full pipeline
#     bash submit_all.sh --skip-fetch    # reuse existing RUWE-clean catalogs
#     RAND_FRACTION=0.1 bash submit_all.sh
#
# After it prints the job IDs:   squeue -u "$USER"
# The final figures land in results/ ; pull them down to update the paper.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

RAND_FRACTION="${RAND_FRACTION:-0.1}"   # matches the original 10% yield sample
RUWE_MAX="${RUWE_MAX:-1.4}"
SKIP_FETCH=0
[ "${1:-}" = "--skip-fetch" ] && SKIP_FETCH=1

mkdir -p logs

# --- choose an env python for the login-node fetch --------------------------
module load conda 2>/dev/null || true
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate optical_reef 2>/dev/null || true
export PATH="$HOME/.conda/envs/optical_reef/bin:$PATH"
PY=python
echo ">> login-node python: $(command -v $PY)"

have_ruwe () {   # $1 = csv path -> 0 if header contains a 'ruwe' column
    [ -f "$1" ] && head -1 "$1" | tr ',' '\n' | grep -qx "ruwe"
}

# --- 0. fetch (login node) --------------------------------------------------
if [ "$SKIP_FETCH" -eq 0 ]; then
    echo ">> [0/5] fetching RUWE-clean catalogs (RUWE < ${RUWE_MAX}) ..."
    $PY fetch_local.py  --rand_fraction "${RAND_FRACTION}" --ruwe_max "${RUWE_MAX}"
    $PY fetch_nearby.py --ruwe_max "${RUWE_MAX}"
else
    echo ">> [0/5] --skip-fetch: reusing existing catalogs"
fi

# hard gate: the cube/sensitivity jobs run on internet-less nodes and will crash
# in catalog.py if the catalog is a pre-RUWE pull. Fail here instead.
for c in gaia_pole_sample.csv gaia_nearby.csv; do
    if ! have_ruwe "$c"; then
        echo "!! $c is missing or has no 'ruwe' column -- re-run without --skip-fetch." >&2
        exit 1
    fi
done
echo ">> catalogs present and RUWE-aware."

# --- 1-2. fine cube -> cube_fine.h5 -----------------------------------------
CUBE=$(sbatch --parsable submit_cube_fine.slurm)
echo ">> [1/5] cube array           : $CUBE"
CUBE_COMB=$(sbatch --parsable --dependency=afterok:"$CUBE" combine_cube_fine.slurm)
echo ">> [2/5] cube combine         : $CUBE_COMB  (after $CUBE)"

# --- 3-4. sensitivity -> summary.json ---------------------------------------
SENS=$(sbatch --parsable submit_sensitivity_array.slurm)
echo ">> [3/5] sensitivity array    : $SENS"
SENS_COMB=$(sbatch --parsable --dependency=afterok:"$SENS" combine_sensitivity.slurm)
echo ">> [4/5] sensitivity combine  : $SENS_COMB  (after $SENS)"

# --- 5. figures (needs cube_fine.h5 AND summary.json) -----------------------
FIG=$(sbatch --parsable --dependency=afterok:"$CUBE_COMB":"$SENS_COMB" submit_figures.slurm)
echo ">> [5/5] figures              : $FIG  (after $CUBE_COMB & $SENS_COMB)"

echo ""
echo "============================================================"
echo " Submitted. Watch:   squeue -u \"$USER\""
echo " Figures appear in results/ when job $FIG finishes."
echo "============================================================"
