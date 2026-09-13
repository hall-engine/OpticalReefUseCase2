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
source activate optical_reef

mkdir -p logs

# Fraction of DR3 to pull. 0.1 = full-analysis default (see notes); 0.03 = quick.
FRAC=0.1

# --fetch_retries 250 with 5-min cap ~ keeps trying for the whole 24 h window.
python -u run.py --fetch_only --rand_fraction ${FRAC} --fetch_retries 250

echo "FETCH DONE -> gaia_pole_sample.csv  (now: sbatch submit_cube_test.slurm)"
