#!/bin/bash
#SBATCH --job-name=lfbot_run3
#SBATCH --partition=ayh24_0001
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --array=1-9
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

SOURCES=(wpp wpp wpp wpp wpp wpp dbl dbl dbl)
EPOCHS=(1 2 3 4 5 6 1 2 3)
IDX=$((SLURM_ARRAY_TASK_ID - 1))
SOURCE=${SOURCES[$IDX]}
EPOCH=${EPOCHS[$IDX]}

NWALKERS=24
NSAMPLES=1e5
NCORES=$SLURM_CPUS_PER_TASK
RUNTAG="run3"
RESUME="--resume"

# ---- run3: eps_e fixed at 0.01 instead of the run1 default of 0.1
#            (eps_B stays at its run1 default of 0.1) ----
EXTRA_ARGS="--eps_e 0.01"

mkdir -p logs

module purge
module load gcc/11.4.0
source /home/fs01/mc2923/R/bin/activate

python runSampler.py \
    --mode fitted_R \
    --source "$SOURCE" \
    --epoch "$EPOCH" \
    --nwalkers "$NWALKERS" \
    --nsamples "$NSAMPLES" \
    --ncores "$NCORES" \
    --dir "$RUNTAG" \
    $RESUME \
    $EXTRA_ARGS
