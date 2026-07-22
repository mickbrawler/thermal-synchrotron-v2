#!/bin/bash
#SBATCH --job-name=lfbot_run2
#SBATCH --partition=ayh24_0001
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --array=1-9
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

# ---- map each of the 9 array tasks to a (source, epoch) pair ----
#   task index:  1    2    3    4    5    6    7    8    9
#   source:     wpp  wpp  wpp  wpp  wpp  wpp  dbl  dbl  dbl
#   epoch:       1    2    3    4    5    6    1    2    3
SOURCES=(wpp wpp wpp wpp wpp wpp dbl dbl dbl)
EPOCHS=(1 2 3 4 5 6 1 2 3)
IDX=$((SLURM_ARRAY_TASK_ID - 1))
SOURCE=${SOURCES[$IDX]}
EPOCH=${EPOCHS[$IDX]}

# ---- per-run settings ----
NWALKERS=24
NSAMPLES=1e5
NCORES=$SLURM_CPUS_PER_TASK
RUNTAG="run2"
RESUME="--resume"

# ---- run2: eps_B fixed at 0.01 instead of the run1 default of 0.1 ----
EXTRA_ARGS="--eps_B 0.01"

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

# After all 9 tasks finish, aggregate per source (run separately, not part
# of the array):
#   python runSampler.py --make_evolution_plot --source wpp --dir run2
#   python runSampler.py --make_evolution_plot --source dbl --dir run2
#   python runSampler.py --make_sed_collage --source wpp --dir run2
#   python runSampler.py --make_sed_collage --source dbl --dir run2
