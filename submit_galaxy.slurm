#!/bin/bash
# ============================================================================
# submit_galaxy.slurm  --  pull the tiny all-sky GALACTIC-CONTEXT sample
# (fetch_galaxy_context.py), unattended.
#
# gaia_source only (ra/dec/parallax/G, no join), a very thin random_index
# fraction across a wide distance range (10 pc .. 5 kpc) -- enough to trace the
# thin disk edge-on and place the 300 pc survey sphere in context. Same chunked
# sync fetch as submit_fetch.slurm: ESA first, ARI mirror fallback, --delay pacing.
# Writes gaia_galaxy_context.csv (+ .meta.json) next to the scripts.
#
# Submit:  sbatch submit_galaxy.slurm
# Watch:   tail -f logs/galaxy_<jobid>.out
#
# *** Needs INTERNET (like submit_fetch.slurm). If the probe fails with a
# CONNECTION/DNS error (not a timeout), run on the LOGIN node instead:
#     nohup python fetch_galaxy_context.py --rand_fraction 0.0003 \
#           --workers 4 --delay 1.5 > galaxy.log 2>&1 &
# ============================================================================
#SBATCH --job-name=reef_galaxy
#SBATCH --output=logs/galaxy_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=06:00:00
#SBATCH --partition=main
#SBATCH --account=dbarnhar_1592

module purge
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate optical_reef 2>/dev/null || true
export PATH="$HOME/.conda/envs/optical_reef/bin:$PATH"
echo "using python: $(which python)"

mkdir -p logs

# Very thin sample: 0.0003 ~ 2 chunks, finishes in well under a minute once a
# TAP server answers. Bump if you want a denser cloud.
FRAC=0.0003

python -u fetch_galaxy_context.py --rand_fraction ${FRAC} \
    --workers 4 --delay 1.5 --probe_timeout 60 --chunk_timeout 120
rc=$?

if [ $rc -eq 0 ]; then
    echo "GALAXY FETCH DONE -> gaia_galaxy_context.csv"
    echo "  then locally:  python3 galaxy_context.py   (paper + dark)"
else
    echo "GALAXY FETCH FAILED (rc=${rc}); both TAP servers unreachable -- rerun later."
fi
exit $rc
