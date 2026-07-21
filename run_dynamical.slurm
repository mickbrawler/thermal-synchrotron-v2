#!/bin/bash
#SBATCH --job-name=lfbot_dynamical
#SBATCH --partition=ayh24_0001
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# dynamical mode fits ALL epochs of a source jointly in one chain --
# no job array needed here, just one job per source. Given how much
# more expensive L_ELOS_IHG is (EATS root-solving per epoch per
# likelihood eval), start with a much smaller --nsamples than the
# fitted_R runs and check `--time_only` first, ideally interactively,
# before committing walltime to this.

SOURCE="wpp"
NWALKERS=32
NSAMPLES=1e5
NCORES=$SLURM_CPUS_PER_TASK
RUNTAG="run1"
RESUME="--resume"

mkdir -p logs

module purge
module load gcc/11.4.0
source /home/fs01/mc2923/R/bin/activate

python mcmc_lfbot_fit.py \
    --mode dynamical \
    --source "$SOURCE" \
    --nwalkers "$NWALKERS" \
    --nsamples "$NSAMPLES" \
    --ncores "$NCORES" \
    --dir "$RUNTAG" \
    $RESUME
