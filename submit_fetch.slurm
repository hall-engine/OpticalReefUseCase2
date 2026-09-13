#!/bin/bash
# ============================================================================
# submit_fetch.slurm  --  pull the Gaia DR3 catalog, unattended (fetch_local.py).
#
# Uses the CHUNKED sync fetch: the gaia_source x astrophysical_parameters join is
# split into many small random_index chunks (~1000 rows / ~9 s each) and
# concatenated -- this is what avoids the archive's statement-timeout that killed
# the single big query. Tries the ESA archive first, falls back to the ARI mirror,
# waits/re-probes if both are down, so you can submit it and forget it.
# Writes gaia_pole_sample.csv (+ .meta.json) next to the scripts.
#
# Submit:  sbatch submit_fetch.slurm
# Watch:   tail -f logs/fetch_<jobid>.out
#
# *** IMPORTANT: this needs INTERNET. ***
# CARC Discovery compute nodes generally have outbound internet, so this should
# work. If the probe fails instantly with a CONNECTION/DNS error (not a timeout),
# your compute nodes are walled off -- run it on the LOGIN node instead:
#     nohup python fetch_local.py --rand_fraction 0.1 > fetch.log 2>&1 &
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
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate optical_reef 2>/dev/null || true
# guarantee the env python even if conda activate misbehaves in batch:
export PATH="$HOME/.conda/envs/optical_reef/bin:$PATH"
echo "using python: $(which python)"

mkdir -p logs

# Full-analysis fraction. 0.1 ~ 333 chunks ~ ~50 min on a healthy mirror.
# (Set 0.03 for the ~15-min, 100-chunk sample that matches the original catalog.)
FRAC=0.1

python -u fetch_local.py --rand_fraction ${FRAC} --probe_timeout 20
rc=$?

if [ $rc -eq 0 ]; then
    echo "FETCH DONE -> gaia_pole_sample.csv  (now: sbatch submit_cube_test.slurm)"
else
    echo "FETCH FAILED (rc=${rc}); both TAP servers were unreachable -- rerun later."
fi
exit $rc
