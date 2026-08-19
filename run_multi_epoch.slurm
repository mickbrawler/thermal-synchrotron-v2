#!/bin/bash
#SBATCH --job-name=lfbot_multiep
#SBATCH --partition=ayh24_0001
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --array=1-2
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

# multi_epoch mode fits ALL epochs of a source jointly in ONE chain -- no
# --epoch needed. Array here just covers both sources (2 tasks), not
# epochs, unlike the fitted_R templates.
SOURCES=(wpp dbl)
SOURCE=${SOURCES[$((SLURM_ARRAY_TASK_ID - 1))]}

NWALKERS=36
NSAMPLES=1e6
NCORES=$SLURM_CPUS_PER_TASK
RUNTAG="run_multiep2"   # bumped from run_multiep1 -- different nwalkers,
                         # nsamples, and wpp's epoch set, not a continuation
RESUME="--resume"

# ---- defaults: free params = a, log10n0, pv_ref, alpha; logR_dec=inf
#      (auto-derived from pv_ref/alpha/earliest epoch); pv_form='decel'.
#      ALWAYS run --time_only first -- this mode is new and unbenchmarked
#      on your actual data/cluster. ----
EXTRA_ARGS=""

mkdir -p logs

module purge
module load gcc/11.4.0
source /home/fs01/mc2923/R/bin/activate

# Recommended first: a quick timing check before committing walltime --
# comment the block below back in, run once, then switch to the real
# submission underneath it.
# python runSampler.py \
#     --mode multi_epoch \
#     --source "$SOURCE" \
#     --nwalkers "$NWALKERS" \
#     --nsamples "$NSAMPLES" \
#     --ncores "$NCORES" \
#     --time_only

python runSampler.py \
    --mode multi_epoch \
    --source "$SOURCE" \
    --nwalkers "$NWALKERS" \
    --nsamples "$NSAMPLES" \
    --ncores "$NCORES" \
    --dir "$RUNTAG" \
    $RESUME \
    $EXTRA_ARGS
