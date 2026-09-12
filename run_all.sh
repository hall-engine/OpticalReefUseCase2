#!/bin/bash
# ============================================================================
# run_all.sh  --  full Optical Reef yield analysis in one go.
#
# Produces EVERYTHING: detection + characterization sweeps (with bootstrap CIs
# and figures), the yield-limit decomposition, and the 3D cloud animations.
# OWA is on by default (config k_owa), so this is the complete physics.
#
# Usage:
#     bash run_all.sh            # local: subsampled, fast (~a few minutes)
#     bash run_all.sh full       # full catalogue, publication settings (slow)
#
# The ONLY difference between local and full is sample size / draw counts /
# frame counts -- the analysis itself is identical.
# ============================================================================
set -e
cd "$(dirname "$0")"
PY=python3
MODE=${1:-local}

if [ "$MODE" = "full" ]; then
    echo "### FULL analysis (whole catalogue) -- this is slow; meant for HPC ###"
    TEST=""              ; STARS=""
    DRAWS="--n_draws 3000"
    BOOT="--bootstrap 1000"
    ADRAWS=800           ; AFRAMES=48 ; ACAP=6000
    LSTARS=""            ; LAP=60
else
    echo "### LOCAL analysis (subsampled, fast) ###"
    TEST="--test"        ; STARS="--max_stars 1500"
    DRAWS="--n_draws 1000"
    BOOT="--bootstrap 500"
    ADRAWS=500           ; AFRAMES=30 ; ACAP=6000
    LSTARS="--max_stars 3000" ; LAP=40
fi

echo ""
echo "===== 1/4  Detection sweep (broadband) ====="
$PY run.py $TEST $STARS $DRAWS $BOOT --mode detection \
    --out_dir results/detection

echo ""
echo "===== 2/4  Characterization sweep (R=70 spectroscopy) ====="
$PY run.py $TEST $STARS $DRAWS $BOOT --mode characterization \
    --out_dir results/characterization

echo ""
echo "===== 3/4  Yield-limit decomposition (IWA / OWA / contrast) ====="
$PY plot_limits.py $LSTARS --n_apertures $LAP

echo ""
echo "===== 4/4  3D local-neighbourhood animations (paper + dark) ====="
$PY animate_cloud3d.py --style both --cap_pc 300 \
    --max_stars $ACAP --n_draws $ADRAWS --n_frames $AFRAMES \
    --out_dir results/animations

echo ""
echo "============================================================"
echo " DONE.  Outputs in results/ :"
echo "   detection/         sweep + figures + bootstrap CIs"
echo "   characterization/  sweep + figures"
echo "   limits/            yield_limits.png"
echo "   animations/        reef_cloud3d_{paper,dark}.{gif,mp4}"
echo "============================================================"
