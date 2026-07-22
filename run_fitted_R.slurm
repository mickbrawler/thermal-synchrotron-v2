#!/bin/bash
#SBATCH --job-name=lfbot_fittedR
#SBATCH --partition=ayh24_0001
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --array=1-3                 # <-- one task per epoch; set to match
                                      #     len(EPOCH_GROUPS[SOURCE]), e.g.
                                      #     1-6 for wpp, 1-3 for dbl
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

# ---- per-job variables ----
SOURCE="dbl"
EPOCH=$SLURM_ARRAY_TASK_ID
NWALKERS=24
NSAMPLES=1e5
NCORES=$SLURM_CPUS_PER_TASK
RUNTAG="run1_dbl"                        # bump this per attempt/config
RESUME="--resume"                    # or "" for a fresh start

mkdir -p logs

# ---- environment ----
module purge
module load gcc/11.4.0
source /home/fs01/mc2923/R/bin/activate

# ---- run ----
# Each array task fits ONE epoch independently (fitted_R mode).
# Re-submitting the same array with RESUME="--resume" will continue any
# task that got killed by walltime, picking up from its saved .h5 chain.
python runSampler.py \
    --mode fitted_R \
    --source "$SOURCE" \
    --epoch "$EPOCH" \
    --nwalkers "$NWALKERS" \
    --nsamples "$NSAMPLES" \
    --ncores "$NCORES" \
    --dir "$RUNTAG" \
    $RESUME

# After ALL array tasks for this source finish, aggregate into the
# parameter-evolution plot with (run separately, not part of the array):
#   python mcmc_lfbot_fit.py --make_evolution_plot --source wpp --dir run1
