#!/bin/bash
# make_all_plots.sh
# Regenerates the evolution+density-profile plots and SED collage for
# both sources, across all 5 run configs. Safe to re-run any time --
# these all just read already-saved data, no resampling happens.
#
# NOTE: SED collage reconstructs the SED using whatever's currently
# passed as fixed params for anything that WASN'T free in that run -- so
# each run's SED collage call below passes the matching override for
# whatever it fixed away from the run1 defaults (eps_B=0.1, eps_e=0.1).
# The evolution/density-profile plots never need this since they're read
# straight from each epoch's saved posterior summary.
#
# Usage: bash make_all_plots.sh
set -e  # stop immediately if any command fails, rather than plowing on

for SOURCE in wpp dbl; do

    # ===== run1 (defaults: eps_B=0.1, eps_e=0.1) =====
    python runSampler.py --make_evolution_plot --source "$SOURCE" --dir run1
    python runSampler.py --make_sed_collage    --source "$SOURCE" --dir run1

    # ===== run2 (eps_B fixed at 0.01) =====
    python runSampler.py --make_evolution_plot --source "$SOURCE" --dir run2
    python runSampler.py --make_sed_collage    --source "$SOURCE" --dir run2 --eps_B 0.01

    # ===== run3 (eps_e fixed at 0.01) =====
    python runSampler.py --make_evolution_plot --source "$SOURCE" --dir run3
    python runSampler.py --make_sed_collage    --source "$SOURCE" --dir run3 --eps_e 0.01

    # ===== run4 (eps_B FREE -- auto-recovered from the fit, no override needed) =====
    python runSampler.py --make_evolution_plot --source "$SOURCE" --dir run4
    python runSampler.py --make_sed_collage    --source "$SOURCE" --dir run4

    # ===== run5 (eps_B FREE + eps_e fixed at 0.01) =====
    python runSampler.py --make_evolution_plot --source "$SOURCE" --dir run5
    python runSampler.py --make_sed_collage    --source "$SOURCE" --dir run5 --eps_e 0.01

done

echo "Done -- all evolution/density-profile/SED-collage plots regenerated for run1-run5."
