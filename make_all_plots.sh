#!/bin/bash
# make_all_plots.sh
# Regenerates the evolution+density-profile plots, SED collage, SED
# overlay, slope-vs-frequency collage, and high-frequency linear-fit
# collage. Safe to re-run any time -- these all just read already-saved
# data, no resampling happens.
#
# USAGE:
#   bash make_all_plots.sh                # everything: both sources, all 5 runs
#   bash make_all_plots.sh wpp             # wpp only, all 5 runs
#   bash make_all_plots.sh wpp run3        # wpp + run3 only
#   bash make_all_plots.sh "" run3         # both sources, run3 only
#                                            (empty "" needed to skip SOURCE
#                                            while still specifying RUN)
#
# NOTE: SED collage/overlay reconstruct the SED using whatever's currently
# passed as fixed params for anything that WASN'T free in that run -- so
# each run below carries the matching override for whatever it fixed away
# from the run1 defaults (eps_B=0.1, eps_e=0.1). The evolution/density-
# profile plots never need this since they're read straight from each
# epoch's saved posterior summary. The slope collage and high-frequency
# collage don't need it either (or any fit at all) -- they're pure data,
# so they come out identical across run1-5; they're just regenerated into
# each run's own plot folder for convenience.

set -e  # stop immediately if any command fails, rather than plowing on

SOURCE_ARG="$1"
RUN_ARG="$2"

if [ -n "$SOURCE_ARG" ]; then
    SOURCES=("$SOURCE_ARG")
else
    SOURCES=(wpp dbl)
fi

# parallel arrays: run name <-> its fixed-param override (matching what
# that run's actual fit used away from the run1 defaults)
RUN_NAMES=(run1 run2 run3 run4 run5)
RUN_EXTRA_ARGS=("" "--eps_B 0.01" "--eps_e 0.01" "" "--eps_e 0.01")

for SOURCE in "${SOURCES[@]}"; do
    for i in "${!RUN_NAMES[@]}"; do
        RUN="${RUN_NAMES[$i]}"
        EXTRA="${RUN_EXTRA_ARGS[$i]}"

        if [ -n "$RUN_ARG" ] && [ "$RUN" != "$RUN_ARG" ]; then
            continue
        fi

        echo "=== $SOURCE / $RUN ==="
        python runSampler.py --make_evolution_plot   --source "$SOURCE" --dir "$RUN"
        python runSampler.py --make_sed_collage      --source "$SOURCE" --dir "$RUN" $EXTRA
        python runSampler.py --make_sed_overlay      --source "$SOURCE" --dir "$RUN" $EXTRA
        python runSampler.py --make_slope_collage    --source "$SOURCE" --dir "$RUN"
        python runSampler.py --make_highfreq_collage --source "$SOURCE" --dir "$RUN"
    done
done

echo "Done -- all evolution/density-profile/SED-collage/SED-overlay/slope-collage/highfreq-fit plots regenerated."
