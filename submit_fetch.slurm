#!/bin/bash
# ============================================================================
# submit_fetch.slurm  --  pull the Gaia DR3 catalog, unattended, with retries.
#
# The archive returns HTTP 500 / statement-timeout when it is overloaded. This
# job just keeps retrying (exponential backoff, capped at 5 min) until it hits a
# healthy window, so you can submit it and walk away. Writes gaia_pole_sample.csv
# next to the scripts, then exits (--fetch_only: no analysis).
#
# Submit:  sbatch submit_fetch.slurm
# Watch:   tail -f logs/fetch_<jobid>.out
#
# *** IMPORTANT: this needs INTERNET. ***
# CARC Discovery compute nodes generally have outbound internet, so this should
# work. If every attempt fails instantly with a CONNECTION/DNS error (not a 500),
# your compute nodes are walled off -- run it on the LOGIN node instead:
#     nohup python run.py --fetch_only --rand_fraction 0.1 > fetch.log 2>&1 &
# ============================================================================
#SBATCH --job-name=reef_fetch
#SBATCH --output=logs/fetch_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=24:00:00
#SBATCH --partition=main
#SBATCH --account=dbarnhar_1592

module purge
module load conda
# robust conda activation for a non-interactive (batch) shell:
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate optical_reef
echo "using python: $(which python)"

mkdir -p logs

# Two-stage fetch: try the full-analysis fraction a few times; if it won't go
# (e.g. the big join keeps hitting the archive's statement timeout), fall back to
# the proven smaller fraction with long retries so it still rides out any overload.
FRAC_MAIN=0.1        ; TRIES_MAIN=10        # ~40 min of attempts
FRAC_FALLBACK=0.03   ; TRIES_FALLBACK=250   # fills the rest of the 24 h window

echo "=== stage 1: trying rand_fraction=${FRAC_MAIN} (${TRIES_MAIN} attempts) ==="
python -u run.py --fetch_only --rand_fraction ${FRAC_MAIN} --fetch_retries ${TRIES_MAIN}
rc=$?

if [ $rc -ne 0 ]; then
    echo "=== stage 1 failed (rc=${rc}); falling back to rand_fraction=${FRAC_FALLBACK} ==="
    python -u run.py --fetch_only --rand_fraction ${FRAC_FALLBACK} --fetch_retries ${TRIES_FALLBACK}
    rc=$?
fi

if [ $rc -eq 0 ]; then
    echo "FETCH DONE -> gaia_pole_sample.csv  (now: sbatch submit_cube_test.slurm)"
else
    echo "FETCH FAILED after both fractions; the archive is likely still down."
fi
exit $rc
