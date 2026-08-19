"""
mcmc_lfbot_fit.py
==================
True MCMC (emcee) fitting of the FM25 thermal+non-thermal synchrotron model
to AT2024wpp ("wpp") and AT2026dbl ("dbl").

Two fitting MODES, corresponding to the two physical pictures in
flux_variables.py:

  * "fitted_R"  -> flux_variables.LOS_IHG_Fitted_R
                   ONE EPOCH at a time, fit independently. R is a free
                   parameter (no assumption tying epochs together
                   dynamically). Fast: no EATS root-solving.
                   Free params: a, log10(R), BG, log10(n0)
                   To build a parameter-evolution-with-time plot, run this
                   mode once per epoch (see EVOLUTION PLOT section below),
                   then aggregate the saved per-epoch summaries.

  * "dynamical" -> flux_variables.L_ELOS_IHG
                   ALL epochs of a source, fit JOINTLY in one chain. R is
                   DERIVED at each epoch from a single shared deceleration
                   law (bG_sh0, alpha) via Shell.py's EATS solve. Physically
                   more constraining, but MUCH slower (Shell.R_EATS_interp
                   does up to ~8000-1e5 scipy.optimize.root_scalar calls per
                   hydrodynamic_variables() call, once per epoch per
                   likelihood evaluation).
                   Free params (shared across all epochs): a, BG0, alpha,
                   log10(n0)

Fixed (not fit) microphysical parameters, in both modes: s, delta, eps_e,
eps_T, p, mu_u, mu_e, k, eps_B -- all editable in the CONTROL PANEL below.

Chains are stored with emcee's HDF5 backend (one .h5 file per run), which
means a run can be safely resumed (RESUME / --resume) after a SLURM job
hits its walltime, and the flat post-burn-in/thinned samples are also
written out as a plain CSV for easy inspection.

USAGE
-----
Local quick test, edit the CONTROL PANEL below then:
    python mcmc_lfbot_fit.py

Everything in the control panel can be overridden from the command line
(handy for SLURM array jobs where you don't want to edit the file per job):
    python mcmc_lfbot_fit.py --source wpp --mode fitted_R --epoch 3 \
        --nwalkers 32 --nsamples 5000 --ncores 24 --dir run1

Timing check before committing to a long run:
    python mcmc_lfbot_fit.py --time_only

After all epochs of a source have been fit in fitted_R mode, build the
parameter-evolution-with-time plot from the saved summaries:
    python mcmc_lfbot_fit.py --make_evolution_plot --source wpp --dir run1
"""

import os
import sys
import time
import glob
import re
import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.cosmology import Planck18

import emcee
from emcee.backends import HDFBackend
import corner

import flux_variables
import Constants as C


# =====================================================================
# CONTROL PANEL -- edit these for a local/interactive run. Any of them
# can be overridden from the command line (see CLI section below), which
# is what you'd do from a SLURM script instead of editing this file.
# =====================================================================

# --- what to run ---
MODE   = "fitted_R"      # "fitted_R"  or  "dynamical"
SOURCE = "wpp"            # "wpp"  or  "dbl"
EPOCH  = 1                 # 1-indexed; only used for MODE="fitted_R".
                           # Can also be "all" to loop over every epoch of
                           # SOURCE in one local invocation (not recommended
                           # for SLURM -- submit one job per epoch there).

# --- sampler settings ---
NWALKERS = 32
NSAMPLES = 5000           # steps PER WALKER (emcee convention), not total
                           # draws. Total posterior draws before burn-in
                           # removal = NWALKERS * NSAMPLES.
NCORES   = 10              # you said htop shows 10 on your machine
SEED     = None

# --- convergence-based early stopping (recommended for long cluster runs
# so you don't burn 1e7 steps if the chain converged after 2e5). Checked
# via the standard emcee autocorrelation-time recipe. ---
CHECK_CONVERGENCE        = True
CONVERGENCE_CHECK_EVERY  = 100     # steps between checks
CONVERGENCE_NTAU         = 50      # require iteration > NTAU * tau
CONVERGENCE_RTOL         = 0.01    # and tau changed by < 1% since last check

# --- fixed (non-fit) microphysical / geometric parameters ---
FIXED_PARAMS = dict(
    s      = 1e5,    # effective range of B-field values B1/B0
    delta  = 1.0,     # index relating n ~ n_hom * B^delta
    eps_e  = 0.1,     # fraction of energy in power-law electrons
    eps_T  = 0.4,     # fraction of energy in thermal electrons
    p      = 3.0,     # power-law electron index
    mu_u   = 0.62,
    mu_e   = 1.18,
    k      = 0.0,     # power-law index for stratified density (0=uniform)
    eps_B  = 0.1,     # fraction of energy in B-field -- FIXED. try 0.01 too
    alpha  = 1.0,     # deceleration index -- INERT/unused for fitted_R
                       # (its signature accepts alpha but never uses it),
                       # meaningful for dynamical mode and the new
                       # multi_epoch mode below (where it's normally FREE)
)

# --- which parameters are FREE in fitted_R mode (edit this to change
# what's fit vs fixed -- e.g. add "log10eps_B" here to fit eps_B instead
# of holding it at FIXED_PARAMS["eps_B"]). Anything not listed here stays
# fixed at its FIXED_PARAMS value. A "log10<name>" entry samples in log10
# space and is converted back to a linear value automatically; anything
# without that prefix (a, BG) is sampled directly. ---
FREE_PARAMS_FITTED_R = ["a", "log10R", "BG", "log10n0"]
# e.g. for eps_B free instead of fixed:
# FREE_PARAMS_FITTED_R = ["a", "log10R", "BG", "log10n0", "log10eps_B"]

# --- which electron populations to include in the model ---
THERM_EL = True
PL_EL    = True

# --- priors for the free parameters (uniform on the given [lo, hi]).
# Only whichever ones are listed in FREE_PARAMS_FITTED_R actually get
# sampled -- the rest of this dict just sits here unused, so it's safe to
# leave entries for parameters you're not currently fitting as free. ---
PRIORS_FITTED_R = dict(
    a          = (0.5, 4.0),
    log10R     = (14.0, 18.0),
    BG         = (0.01, 2.10),
    log10n0    = (-2.0, 6.0),
    # eps_B in (0.001, 0.5) -> log10 space (-3.0, -0.301). The
    # eps_e+eps_B+eps_T<=1 physicality cut (checked dynamically per-sample)
    # will additionally carve out any unphysical corner within this range.
    log10eps_B = (-3.0, -0.301),
    # eps_ratio = eps_e/eps_T sampled directly (see _build_theta_params) --
    # with eps_T=0.4 fixed, this range corresponds to eps_e roughly in
    # (0.0004, 0.5). Adjust via --log10eps_ratio_bounds if you want a
    # different range.
    log10eps_ratio = (-3.0, 0.1),
)
# dynamical mode fits: a, BG0, alpha, log10n0 (shared across all epochs)
# NOTE: alpha bounds are a rough placeholder -- tune these.
PRIORS_DYNAMICAL = dict(
    a       = (0.5, 4.0),
    BG0     = (0.01, 2.0),
    alpha   = (0.0, 3.0),
    log10n0 = (-2.0, 6.0),
)

# --- multi_epoch mode: joint fit across ALL epochs of a source at once,
# using LOS_IHG_MULTI_EPOCH. Like 'dynamical' mode, R(t) is DERIVED (not
# fit directly) from a shared deceleration law -- but via one
# scipy.optimize.root_scalar solve per epoch (through the new R0()/
# radius_implicit_calc() helpers), not the ~8000-100000-per-epoch EATS
# integral Shell.py/L_ELOS_IHG needs -- i.e. this should make joint
# dynamical-style fitting actually practical speed-wise. Free params
# default to a, log10n0, pv_ref (proper velocity at the reference
# epoch/radius -- same ROLE as BG/BG0 before), and alpha (the new
# deceleration index) -- there's no log10R-equivalent free parameter
# since R is derived at every epoch from this shared law. ---
FREE_PARAMS_MULTI_EPOCH = ["a", "log10n0", "pv_ref", "alpha"]

PRIORS_MULTI_EPOCH = dict(
    a       = (0.5, 4.0),
    log10n0 = (-2.0, 6.0),
    pv_ref  = (0.01, 2.10),
    alpha   = (0.1, 3.0),    # placeholder -- tune this
)

# Reference radius (log10 cm) at which pv_ref is defined. float('inf')
# (default) auto-derives it from pv_ref/alpha and the EARLIEST epoch's own
# time -- the natural choice matching "R is derived, not fit". Set to a
# specific log10(cm) value instead via --multi_logR_dec to anchor pv_ref
# at a radius you choose rather than the first epoch.
MULTI_EPOCH_LOGR_DEC = float("inf")

# 'pl' (pure power-law deceleration, no coasting phase) or 'decel'
# (includes an initial coasting phase before deceleration sets in) --
# your call which is more physically appropriate; defaulting to 'decel'
# as the more general choice. Override via --multi_pv_form.
MULTI_EPOCH_PV_FORM = "decel"

# --- data handling ---
SNR_THRESHOLD       = 3.0
FLUX_ERR_FLOOR_FRAC = 0.10    # 10% error floor

# --- SED plot axis padding: how far beyond the data's own min/max (in log10
# decades) the displayed axis and evaluated fit curve extend on each side.
# 0.15 dex ~= factor of 1.4 in frequency/flux -- "a little" beyond the data,
# not a fixed multi-decade span. Same value drives both nu_grid's range and
# the axis limits, so the lines always reach the plot edges (no dead margin).
SED_PAD_DEX = 0.15

# --- optional chi2 weighting for points past an epoch's empirical SED
# peak (the data's own peak flux, not anything model-dependent). Off by
# default -- turn on with --chi2_weight_post_peak or set True here. Only
# affects the FIT (log_prob_fitted_R); plots/eflux/error bars elsewhere
# are completely unaffected, this never touches the actual stored data. ---
CHI2_WEIGHT_POST_PEAK = False
CHI2_POST_PEAK_WEIGHT = 3.0   # chi2 multiplier applied to those points
                                # when the above is True -- tune via
                                # --chi2_post_peak_weight

# --- output ---
OUTDIR  = "./mcmc_output"
RUN_TAG = "1"            # subdirectory label, a la your old SLURM $DIR;
                           # bump this per attempt/config so runs don't clobber
RESUME  = False            # if True, continue an existing chain (same
                           # nwalkers/ndim) instead of starting fresh --
                           # this is what "FOLLOWUP" did in your old script

# --- one-off utility actions (normally left False; CLI flags flip these) ---
TIME_ONLY           = False
MAKE_EVOLUTION_PLOT = False
MAKE_KNOB_PLOT       = False
MAKE_ALL_KNOB_PLOTS  = False
MAKE_SED_COLLAGE     = False
MAKE_SED_OVERLAY     = False
OVERLAY_YLABEL       = None  # set to a string to show a y-axis label on
                              # the SED overlay plot; default is no label
MAKE_SLOPE_COLLAGE   = False
MAKE_HIGHFREQ_COLLAGE = False
REPLOT               = False

# --- "turning knobs" SED sensitivity plot (fitted_R model only) ---
# For each of the 11 non-mu params (the 4 free fitted_R params + the 7
# fixed microphysical/geometric params, excluding mu_u/mu_e), holds every
# other parameter at KNOB_CENTER_* and varies just that one across
# KNOB_BOUNDS, plotted as a 3x4 collage (top-left = unchanged baseline).
# Center values default to the midpoint of each free param's prior and
# the current FIXED_PARAMS values; pass --knob_from_summary to instead
# center the free params on a real epoch's MAP fit (also picks up that
# epoch's T automatically).
KNOB_SOURCE          = "wpp"     # only used for z / d_L (and T if no
                                  # --knob_from_summary given)
KNOB_T               = 50.0      # observer-frame days, used unless
                                  # --knob_from_summary supplies its own T
KNOB_FROM_SUMMARY    = None      # path to a "*_summary.csv"; None = use
                                  # prior midpoints for the free params
KNOB_N_VALUES        = 5
KNOB_VARIANT         = "combined"  # "combined" / "pl_only" / "therm_only"
# combined:   uses THERM_EL/PL_EL as set above (or --therm_el/--pl_el)
# pl_only:    forces therm_el=False, pl_el=True  (isolates the PL population)
# therm_only: forces therm_el=True, pl_el=False  (isolates the thermal population)
# Note eps_T still affects the pl_only variant (it sets Theta, which sets
# gamma_m -- where the PL population starts) -- only eps_e is truly inert
# in the therm_only variant.

# (lo, hi) sweep range for each dial. Free params reuse PRIORS_FITTED_R
# below (after that dict is defined); fixed-param ranges are physically
# motivated guesses -- these are yours to tune.
KNOB_BOUNDS_FIXED = dict(
    s      = (1e2, 1e6),
    delta  = (0.0, 3.0),
    eps_e  = (0.01, 0.5),
    eps_T  = (0.05, 0.6),
    p      = (2.1, 4.0),
    k      = (0.0, 2.0),
    eps_B  = (0.001, 0.5),
)
KNOB_LOGSPACE_FIXED = dict(
    s=True, delta=False, eps_e=True, eps_T=False, p=False, k=False, eps_B=True,
)
KNOB_LOGSPACE_FREE = dict(a=False, log10R=False, BG=False, log10n0=False)


# =====================================================================
# SOURCE / EPOCH DEFINITIONS
# =====================================================================

# EDIT to match your machine / where you keep the raw data files.
_FILE_DIR = "/home/fs01/mc2923/thermal-synchrotron-v2/sourceData/"

T0_MJD_WPP = 60579.37093750
T0_MJD_DBL = 61085.34930556
# NOTE: the current dbl() loader below reads a pre-processed "dbl.txt"
# that already has a "dt(days)" column (i.e. T0 already subtracted), so
# T0_MJD_DBL isn't used yet. It's defined here so it's ready to go if/when
# you switch dbl to raw MJD-based subband files like wpp's -- let me know
# and I'll wire up an analogous loader once I know those filenames.

SOURCE_INFO = {
    "wpp": {"z": 0.0868},
    "dbl": {"z": 0.19},
}

# Epoch bounds in rest-frame days. Comment out epochs to skip.
EPOCH_GROUPS = {
    "wpp": [
        (25,  35),
        (35,  50),
        (60,  75),
        (100, 110),
        (110, 120),
        (170, 180),
    ],
    "dbl": [
        (9,   12),
        (15,  18),
        (22,  27),
    ],
}


def _a_upper_bound(p, delta):
    """Physical validity bound on `a` from flux_variables.P()'s docstring:
    the model requires 1/2 < a < (p+3)/2 + delta."""
    return (p + 3.0) / 2.0 + delta


# =====================================================================
# DISPLAY FORMATTING (titles, math-style axis labels, natural plot ranges)
# =====================================================================

SOURCE_DISPLAY_NAMES = {"wpp": "AT2024wpp", "dbl": "AT2026dbl"}

# LaTeX-ish labels for anything that shows up on a plot axis/title.
LABEL_MATH = {
    "a":       r"$a$",
    "log10R":  r"$\log_{10}(R\ [\mathrm{cm}])$",
    "BG":      r"$\Gamma\beta$",
    "log10n0": r"$\log_{10}(n_0\ [\mathrm{cm}^{-3}])$",
    "BG0":     r"$\Gamma\beta_0$",
    "alpha":   r"$\alpha$",
    "s":       r"$s$",
    "delta":   r"$\delta$",
    "eps_e":   r"$\epsilon_e$",
    "eps_T":   r"$\epsilon_T$",
    "eps_B":   r"$\epsilon_B$",
    "p":       r"$p$",
    "k":       r"$k$",
    "mu_u":    r"$\mu_u$",
    "mu_e":    r"$\mu_e$",
}


def pretty_label(lab):
    return LABEL_MATH.get(lab, lab)


def pretty_title(source, mode, epoch=None):
    name = SOURCE_DISPLAY_NAMES.get(source, source)
    if mode == "fitted_R":
        return f"{name} Epoch {epoch}"
    if mode == "multi_epoch":
        return f"{name} Joint Multi-Epoch Fit (All Epochs)"
    return f"{name} Joint Fit (All Epochs)"


def dlnF_dlnnu(F_nu, nu):
    """Numerical local spectral index lambda = d ln(F_nu) / d ln(nu).

    A log-log derivative, so it's invariant to whatever normalization
    F_nu is in (Jy, cgs flux density, or L_nu) -- only the shape matters.

    Parameters
    ----------
    F_nu : array
        Specific flux or luminosity values (any consistent normalization).
    nu : array
        Frequency (Hz), same length as F_nu.

    Returns
    -------
    lognu : array
        log(nu), same ordering as input.
    log_diff : array
        Local spectral index at each frequency.
    """
    logF = np.log(np.abs(F_nu))
    lognu = np.log(nu)
    log_diff = np.gradient(logF, lognu)
    return lognu, log_diff


def natural_xy_limits(nu_grid, curves, freq_data=None, flux_data=None,
                       pad_dex=0.3, thresh_frac=1e-3):
    """Frequency/flux axis limits sized to where the model curves (and
    data, if given) actually have signal, padded a bit ('a little to the
    left and right') rather than a fixed generic multi-decade span."""
    all_curves = np.vstack([np.atleast_2d(c) for c in curves])
    peak = np.nanmax(all_curves)
    if not np.isfinite(peak) or peak <= 0:
        return (nu_grid.min(), nu_grid.max()), (1e-3, 1.0)

    significant = np.any(all_curves > thresh_frac * peak, axis=0)
    if np.any(significant):
        nu_lo, nu_hi = nu_grid[significant].min(), nu_grid[significant].max()
    else:
        nu_lo, nu_hi = nu_grid.min(), nu_grid.max()

    if freq_data is not None and len(freq_data):
        nu_lo = min(nu_lo, np.min(freq_data))
        nu_hi = max(nu_hi, np.max(freq_data))

    log_lo, log_hi = np.log10(nu_lo), np.log10(nu_hi)
    xlim = (10 ** (log_lo - pad_dex), 10 ** (log_hi + pad_dex))

    y_source = all_curves[:, significant] if np.any(significant) else all_curves
    y_source = y_source[y_source > 0]
    y_lo = np.nanmin(y_source) if y_source.size else peak * thresh_frac
    y_hi = peak
    if flux_data is not None and len(flux_data):
        y_lo = min(y_lo, np.min(flux_data))
        y_hi = max(y_hi, np.max(flux_data))
    log_ylo, log_yhi = np.log10(y_lo), np.log10(y_hi)
    ylim = (10 ** (log_ylo - pad_dex), 10 ** (log_yhi + pad_dex))
    return xlim, ylim


def parse_config_txt_fixed(path):
    """Parse the 'Fixed (non-fit) parameters:' section of a saved
    *_config.txt back into a {name: float} dict, so callers can check for
    an actual mismatch against currently-passed FIXED_PARAMS/CLI overrides
    rather than just printing a generic 'go check this yourself' note."""
    fixed = {}
    in_section = False
    with open(path) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped == "Fixed (non-fit) parameters:":
                in_section = True
                continue
            if in_section:
                if not stripped or stripped.startswith(("therm_el", "pl_el")):
                    break
                if "=" in stripped:
                    k, v = stripped.split("=", 1)
                    try:
                        fixed[k.strip()] = float(v.strip())
                    except ValueError:
                        pass
    return fixed


def check_fixed_params_match(config_path, current_fixed, tol=1e-8):
    """Compare a saved run's actual fixed params (from its *_config.txt)
    against what's currently passed in. Returns a list of human-readable
    mismatch strings (empty if everything matches or the file is missing/
    unparseable, since in that case there's nothing to compare against)."""
    if not os.path.exists(config_path):
        return None  # can't verify -- caller should say so explicitly
    try:
        saved = parse_config_txt_fixed(config_path)
    except Exception:
        return None
    mismatches = []
    for k, saved_v in saved.items():
        cur_v = current_fixed.get(k)
        if cur_v is None or abs(cur_v - saved_v) > tol * max(1.0, abs(saved_v)):
            mismatches.append(f"{k}: run used {saved_v}, currently passing {cur_v}")
    return mismatches


def save_run_config_txt(plots_dir, tag, fixed, priors, therm_el, pl_el,
                        free_labels=None, chi2_weight_post_peak=None,
                        chi2_post_peak_weight=None):
    free_keys = set()
    if free_labels:
        free_keys = {(lab[5:] if lab.startswith("log10") else lab)
                     for lab in free_labels}
        if "eps_ratio" in free_keys:
            # eps_e is DERIVED from the free eps_ratio (=eps_e/eps_T) and
            # eps_T, not actually fixed -- don't list it as such below.
            free_keys.add("eps_e")

    lines = [f"Configuration for {tag}", "=" * (len(tag) + 15), "",
             "Fixed (non-fit) parameters:"]
    for k, v in fixed.items():
        if k in free_keys:
            continue  # this one is free for this run -- listed below instead
        lines.append(f"  {k:8s} = {v}")
    lines += ["", "therm_el = " + str(therm_el), "pl_el    = " + str(pl_el)]
    if chi2_weight_post_peak is not None:
        lines.append(f"chi2_weight_post_peak = {chi2_weight_post_peak}"
                     + (f" (factor={chi2_post_peak_weight})" if chi2_weight_post_peak else ""))
    lines += ["", "Free parameters (uniform priors):"]
    if free_labels:
        for lab in free_labels:
            lo, hi = priors[lab]
            lines.append(f"  {lab:12s} : ({lo}, {hi})")
    else:
        for k, (lo, hi) in priors.items():
            lines.append(f"  {k:8s} : ({lo}, {hi})")
    with open(os.path.join(plots_dir, f"{tag}_config.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


# =====================================================================
# DATA LOADING
# =====================================================================

# --- epoch-specific ALMA subband -> full-band swaps -----------------------
# Per-subband S/N was too sparse at these approximate frequencies (GHz) in
# these epochs to be worth using subbanded; use the corresponding full-band
# point(s) from wppALMA.txt instead. Matching is by epoch time-window +
# frequency within ALMA_SUBBAND_EXCLUDE_TOL_FRAC of the target -- adjust the
# tolerance if the printed match counts don't look right on your first run.
ALMA_SUBBAND_EXCLUDE_FREQ_GHZ = {
    3: [200.0, 400.0],   # epoch 3: drop the ~2e11 and ~4e11 Hz subband groups
    4: [90.0],           # epoch 4: drop the ~9e10 Hz subband group
}
ALMA_SUBBAND_EXCLUDE_TOL_FRAC = 0.25

# --- non-detections to keep in the fit as pseudo-detections ---------------
# Included in the fit's likelihood with flux = SNR_THRESHOLD x fluxErr (the
# 3-sigma upper limit value) and error = fluxErr (same error-floor logic as
# everything else downstream in get_epoch_data) -- but still DISPLAYED as
# an upper-limit triangle in plots, not a normal detection dot, since it's
# still fundamentally a non-detection.
VLA_NONDET_PROMOTE_EPOCHS = {6}

# --- non-detections to drop entirely -- not in the fit, never plotted ----
VLA_NONDET_IGNORE_EPOCHS = {5}

# --- per-epoch high-frequency cutoff (Hz) for the high-freq linear-fit
# collage: only points ABOVE this frequency get included in that epoch's
# linear fit (in log-log space) -- used to isolate the optically-thin
# spectral slope where it's cleanly sampled. ---
HIGH_FREQ_THRESHOLD_HZ = {
    "wpp": {1: 1.5e11, 2: 1.5e11, 3: 1.5e11, 4: 3e10, 5: 5e9, 6: 1.3e10},
    "dbl": {1: 1.5e11, 2: 1.5e11, 3: 1.5e11},
}


def _load_wpp_raw():
    days, freq, flux, fluxErr, det, source = [], [], [], [], [], []

    alma_path = os.path.join(_FILE_DIR, "wppALMA_subbands.txt")
    with open(alma_path) as fh:
        header_seen = False
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if not header_seen:
                header_seen = True
                continue
            cols = line.split()
            if len(cols) < 7:
                continue
            days.append(float(cols[6]) - T0_MJD_WPP)
            freq.append(float(cols[0]))
            flux.append(float(cols[2]) * 1e-3)
            fluxErr.append(float(cols[3]) * 1e-3)
            det.append(float(cols[5]) > SNR_THRESHOLD)
            source.append("alma_subbands")

    vla_path = os.path.join(_FILE_DIR, "wppVLA_subbands_epoch1_2_SbandIgnore.dat")
    with open(vla_path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) < 8:
                continue
            days.append(float(cols[7]) - T0_MJD_WPP)
            freq.append(float(cols[0]))
            flux.append(float(cols[3]) * 1e-3)
            fluxErr.append(float(cols[4]) * 1e-3)
            det.append(float(cols[6]) > SNR_THRESHOLD)
            source.append("vla_subbands")

    nondets_path = os.path.join(_FILE_DIR, "wppVLA_nondets_SbandChange.txt")
    with open(nondets_path) as fh:
        header_seen = False
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t") if "\t" in line else line.split(None, 11)
            if not header_seen:
                header_seen = True
                continue
            if len(cols) < 7:
                continue
            try:
                days.append(float(cols[1]))
                freq.append(float(cols[2]))
                flux.append(float(cols[3]) * 1e-3)
                fluxErr.append(float(cols[4]) * 1e-3)
                det.append(bool(int(float(cols[6]))))
                source.append("vla_nondets")
            except ValueError:
                continue

    # ALMA full-band values -- same column layout as the subbands file
    # (freq, spws, flux, ferr, rms, snr, mjd). Loaded here but NOT used
    # everywhere by default; _apply_wpp_alma_swap() below scopes their use
    # to only the specific epoch/frequency combos being swapped in, to
    # avoid double-counting against good subbands elsewhere.
    fullband_path = os.path.join(_FILE_DIR, "wppALMA.txt")
    if os.path.exists(fullband_path):
        with open(fullband_path) as fh:
            header_seen = False
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if not header_seen:
                    header_seen = True
                    continue
                cols = line.split()
                if len(cols) < 7:
                    continue
                days.append(float(cols[6]) - T0_MJD_WPP)
                freq.append(float(cols[0]))
                flux.append(float(cols[2]) * 1e-3)
                fluxErr.append(float(cols[3]) * 1e-3)
                det.append(float(cols[5]) > SNR_THRESHOLD)
                source.append("alma_fullband")
    else:
        print(f"    WARNING: {fullband_path} not found -- ALMA subband/"
              f"full-band swap for epochs {list(ALMA_SUBBAND_EXCLUDE_FREQ_GHZ)} "
              f"will NOT be applied (falling back to raw subbands there).")

    order = np.argsort(days)
    raw = {
        "days":    np.array(days)[order],
        "freq":    np.array(freq)[order],
        "flux":    np.array(flux)[order],
        "fluxErr": np.array(fluxErr)[order],
        "det":     np.array(det, dtype=bool)[order],
        "source":  np.array(source)[order],
        # tracks points that ARE included in the fit (det=True) but should
        # still be DISPLAYED as upper limits (e.g. a promoted non-detection)
        # rather than a normal detection marker. False for everything else.
        "is_upper_limit": np.zeros(len(order), dtype=bool),
    }

    raw = _apply_wpp_alma_swap(raw)
    raw = _apply_wpp_nondet_ignore(raw)
    raw = _apply_wpp_nondet_promotion(raw)
    return raw


def _apply_wpp_alma_swap(raw):
    """Drop specific ALMA subband groups in specific epochs (per
    ALMA_SUBBAND_EXCLUDE_FREQ_GHZ) and swap in the matching full-band
    point(s) for THOSE epochs only -- full-band points are excluded
    everywhere else by default to avoid double-counting against otherwise-
    good subbands."""
    z = SOURCE_INFO["wpp"]["z"]
    t_rest = raw["days"] / (1 + z)
    is_subband = raw["source"] == "alma_subbands"
    is_fullband = raw["source"] == "alma_fullband"

    keep = np.ones(len(raw["days"]), dtype=bool)
    keep &= ~is_fullband  # excluded by default everywhere

    for epoch_idx, targets_ghz in ALMA_SUBBAND_EXCLUDE_FREQ_GHZ.items():
        t_lo, t_hi = EPOCH_GROUPS["wpp"][epoch_idx - 1]
        in_epoch = (t_rest >= t_lo) & (t_rest <= t_hi)
        for target in targets_ghz:
            near = np.abs(raw["freq"] / target - 1.0) < ALMA_SUBBAND_EXCLUDE_TOL_FRAC
            drop_sub = in_epoch & is_subband & near
            add_full = in_epoch & is_fullband & near
            keep &= ~drop_sub
            keep |= add_full
            print(f"    [wpp epoch {epoch_idx}] near {target:.0f} GHz: "
                  f"dropping {drop_sub.sum()} ALMA subband point(s), "
                  f"adding {add_full.sum()} full-band point(s)")

    return {k: v[keep] for k, v in raw.items()}


def _apply_wpp_nondet_promotion(raw):
    """For epochs in VLA_NONDET_PROMOTE_EPOCHS, convert any non-detection
    there into a pseudo-detection: flux = SNR_THRESHOLD x fluxErr (the
    3-sigma upper-limit value), det=True -- so it participates in the fit
    like any other point. Also flagged is_upper_limit=True so plotting
    code can still render it as an upper limit (dimmed downward triangle)
    rather than a normal detection marker, even though it's now det=True.
    Error stays as fluxErr; the existing error-floor logic in
    get_epoch_data applies to it the same as any other point."""
    if not VLA_NONDET_PROMOTE_EPOCHS:
        return raw
    z = SOURCE_INFO["wpp"]["z"]
    t_rest = raw["days"] / (1 + z)
    det = raw["det"].copy()
    flux = raw["flux"].copy()
    is_upper_limit = raw["is_upper_limit"].copy()

    for epoch_idx in VLA_NONDET_PROMOTE_EPOCHS:
        t_lo, t_hi = EPOCH_GROUPS["wpp"][epoch_idx - 1]
        in_epoch = (t_rest >= t_lo) & (t_rest <= t_hi)
        hit = in_epoch & ~raw["det"]
        n = int(hit.sum())
        if n:
            flux[hit] = SNR_THRESHOLD * raw["fluxErr"][hit]
            det[hit] = True
            is_upper_limit[hit] = True
        print(f"    [wpp epoch {epoch_idx}] promoting {n} non-detection(s) "
              f"to pseudo-detections (flux = {SNR_THRESHOLD}x sigma)")

    raw2 = dict(raw)
    raw2["det"] = det
    raw2["is_upper_limit"] = is_upper_limit
    raw2["flux"] = flux
    return raw2


def _apply_wpp_nondet_ignore(raw):
    """For epochs in VLA_NONDET_IGNORE_EPOCHS, drop non-detection rows
    entirely -- not in the fit, never plotted anywhere (unlike the default
    behavior for a non-detection, which excludes it from the fit but still
    shows it as a dimmed upper-limit triangle)."""
    if not VLA_NONDET_IGNORE_EPOCHS:
        return raw
    z = SOURCE_INFO["wpp"]["z"]
    t_rest = raw["days"] / (1 + z)
    drop = np.zeros(len(raw["days"]), dtype=bool)

    for epoch_idx in VLA_NONDET_IGNORE_EPOCHS:
        t_lo, t_hi = EPOCH_GROUPS["wpp"][epoch_idx - 1]
        in_epoch = (t_rest >= t_lo) & (t_rest <= t_hi)
        hit = in_epoch & ~raw["det"]
        print(f"    [wpp epoch {epoch_idx}] dropping {int(hit.sum())} "
              f"non-detection(s) entirely (never used, never plotted)")
        drop |= hit

    keep = ~drop
    return {k: v[keep] for k, v in raw.items()}


def _load_dbl_raw():
    data = pd.read_csv(os.path.join(_FILE_DIR, "dbl.txt"), sep=r"\s+")
    data["flux(mJy)"] *= 1e-3
    data["ferr(mJy)"] *= 1e-3
    return {
        "days":    data["dt(days)"].values,
        "freq":    data["nu(GHz)"].values,
        "flux":    data["flux(mJy)"].values,
        "fluxErr": data["ferr(mJy)"].values,
        "det":     np.array(data["det"].values, dtype=bool),
        "is_upper_limit": np.zeros(len(data), dtype=bool),
    }


_raw_cache = {}


def _get_raw(source):
    if source not in _raw_cache:
        _raw_cache[source] = _load_wpp_raw() if source == "wpp" else _load_dbl_raw()
    return _raw_cache[source]


def get_epoch_data(source, epoch_idx):
    """epoch_idx is 1-indexed, matching EPOCH_GROUPS[source]."""
    raw = _get_raw(source)
    z = SOURCE_INFO[source]["z"]
    d_L = Planck18.luminosity_distance(z).cgs.value

    t_lo, t_hi = EPOCH_GROUPS[source][epoch_idx - 1]
    t_rest = raw["days"] / (1 + z)
    in_epoch = (t_rest >= t_lo) & (t_rest <= t_hi)
    mask = in_epoch & raw["det"]

    T = float(np.mean(raw["days"][mask]))
    flux_ep = raw["flux"][mask]
    eflux_ep = np.maximum(raw["fluxErr"][mask], FLUX_ERR_FLOOR_FRAC * flux_ep)
    freq_hz = raw["freq"][mask] * 1e9
    # True for any of the above (fit-participating) points that should
    # still be DISPLAYED as an upper limit (e.g. a promoted non-detection)
    # rather than a normal detection marker.
    is_upper_limit_ep = raw["is_upper_limit"][mask]

    # Non-detections in the same epoch window -- NOT included in the fit,
    # only for display. Plotted at SNR_THRESHOLD x their RMS noise
    # (fluxErr), not their raw (possibly noisy/insignificant) measured
    # flux, per standard upper-limit convention.
    nondet_mask = in_epoch & ~raw["det"]
    freq_nondet_hz = raw["freq"][nondet_mask] * 1e9
    upper_limit_nondet = SNR_THRESHOLD * raw["fluxErr"][nondet_mask]

    # actual rest-frame time span/mean of the data points used (distinct
    # from the EPOCH_GROUPS bin edges) -- used for SED collage titles
    t_rest_ep = t_rest[mask]
    if len(t_rest_ep):
        t_rest_min = float(np.min(t_rest_ep))
        t_rest_max = float(np.max(t_rest_ep))
        t_rest_mean = float(np.mean(t_rest_ep))
    else:
        # genuinely zero detections in this epoch -- don't crash, just
        # report NaN for the time-span fields (T above will also be NaN)
        t_rest_min = t_rest_max = t_rest_mean = float("nan")

    return dict(freq=freq_hz, flux=flux_ep, eflux=eflux_ep, T=T, z=z, d_L=d_L,
                epoch=epoch_idx, source=source, is_upper_limit=is_upper_limit_ep,
                freq_nondet=freq_nondet_hz, upper_limit_nondet=upper_limit_nondet,
                t_rest_min=t_rest_min, t_rest_max=t_rest_max,
                t_rest_mean=t_rest_mean)


def get_all_epochs(source):
    n_ep = len(EPOCH_GROUPS[source])
    return [get_epoch_data(source, ep) for ep in range(1, n_ep + 1)]


# =====================================================================
# PRIORS
# =====================================================================

def log_prior_box(theta, labels, bounds):
    for val, lab in zip(theta, labels):
        lo, hi = bounds[lab]
        if not (lo < val < hi):
            return -np.inf
    return 0.0


def check_physicality(fixed, free_labels=()):
    """eps_e + eps_B + eps_T <= 1 must hold (fraction of energy budget).
    Any of these that are FREE parameters (in free_labels) get checked
    dynamically per-sample in log_prob instead -- skip them here rather
    than judging by whatever placeholder value sits in FIXED_PARAMS."""
    free_keys = {(lab[5:] if lab.startswith("log10") else lab) for lab in free_labels}
    eps_e_free = "eps_e" in free_keys or "eps_ratio" in free_keys
    eps_e = 0.0 if eps_e_free else fixed["eps_e"]
    eps_B = 0.0 if "eps_B" in free_keys else fixed["eps_B"]
    eps_T = 0.0 if "eps_T" in free_keys else fixed["eps_T"]
    total = eps_e + eps_B + eps_T
    if total > 1.0:
        raise ValueError(
            f"eps_e + eps_B + eps_T = {total:.3f} > 1.0 -- not physical "
            f"(checked over the currently-FIXED subset only; free ones are "
            f"checked per-sample during sampling). Adjust FIXED_PARAMS."
        )


# =====================================================================
# LIKELIHOODS + LOG-PROB (module level, for picklability with mp spawn)
# =====================================================================

PARAM_LABELS_DYNAMICAL = ["a", "BG0", "alpha", "log10n0"]


def _build_theta_params(theta, labels, fixed):
    """Merge FIXED_PARAMS with the free parameters from theta into one
    physical-parameter dict. A label prefixed with 'log10' is converted to
    its linear value under that key (e.g. 'log10eps_B' -> params['eps_B']
    = 10**theta_value), overriding whatever FIXED_PARAMS had for it.

    Special case: 'log10eps_ratio' is the RATIO eps_e/eps_T sampled
    directly as one parameter (not eps_e itself) -- this is the actual
    quantity that enters the flux calculation as the PL amplitude scaling
    when GRB_convention=False (which is always the case in this pipeline).
    If present, eps_e is DERIVED from it and whatever eps_T currently is
    (fixed or itself free), overriding any eps_e that FIXED_PARAMS had.
    """
    params = dict(fixed)
    for lab, val in zip(labels, theta):
        if lab.startswith("log10"):
            params[lab[5:]] = 10.0 ** val
        else:
            params[lab] = val
    if "eps_ratio" in params:
        params["eps_e"] = params["eps_ratio"] * params["eps_T"]
    return params


def _fnu_fitted_R(theta, labels, freq, T, z, d_L, fixed, therm_el, pl_el):
    p = _build_theta_params(theta, labels, fixed)
    Lnu = flux_variables.LOS_IHG_Fitted_R(
        freq, p["s"], p["a"], p["delta"], p["R"], T, p["n0"],
        p["eps_e"], p["eps_B"], p["eps_T"], p["p"],
        p["mu_u"], p["mu_e"], p["BG"], p["k"], p["alpha"], d_L, z,
        therm_el=therm_el, pl_el=pl_el,
    )
    return Lnu / (4.0 * np.pi * d_L ** 2) / C.Jy


def log_prob_fitted_R(theta, freq, flux, eflux, T, z, d_L, fixed,
                       therm_el, pl_el, bounds, labels, weights):
    lp = log_prior_box(theta, labels, bounds)
    if not np.isfinite(lp):
        return -np.inf

    p = _build_theta_params(theta, labels, fixed)

    if not (0.5 < p["a"] < _a_upper_bound(p["p"], p["delta"])):
        return -np.inf
    # Dynamic physicality check -- matters whenever eps_e/eps_B/eps_T are
    # free (their FIXED_PARAMS placeholder doesn't apply per-sample).
    if p["eps_e"] + p["eps_B"] + p["eps_T"] > 1.0:
        return -np.inf

    try:
        Fnu_model = flux_variables.LOS_IHG_Fitted_R(
            freq, p["s"], p["a"], p["delta"], p["R"], T, p["n0"],
            p["eps_e"], p["eps_B"], p["eps_T"], p["p"],
            p["mu_u"], p["mu_e"], p["BG"], p["k"], p["alpha"], d_L, z,
            therm_el=therm_el, pl_el=pl_el,
        ) / (4.0 * np.pi * d_L ** 2) / C.Jy
        chi2 = np.sum(weights * ((flux - Fnu_model) / eflux) ** 2)
        ll = -0.5 * chi2
    except Exception:
        return -np.inf

    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def compute_post_peak_weights(freq, flux, enabled, weight_factor):
    """Per-point chi2 weight array: 1.0 everywhere, except weight_factor
    for any point at a frequency ABOVE the epoch's own empirical peak
    flux (i.e. the observed data's peak, not anything model-dependent --
    this must stay fixed throughout sampling, not shift around with theta,
    or the likelihood wouldn't be well-defined as a fixed target
    distribution). No-op (all 1.0) if `enabled` is False."""
    weights = np.ones_like(freq, dtype=float)
    if enabled and len(freq) > 0:
        peak_freq = freq[np.argmax(flux)]
        weights[freq > peak_freq] = weight_factor
    return weights


def _fnu_dynamical(theta, freq, T, z, d_L, fixed, therm_el, pl_el):
    a, BG0, alpha, log10n0 = theta
    n0 = 10.0 ** log10n0
    Lnu = flux_variables.L_ELOS_IHG(
        freq, fixed["s"], a, fixed["delta"], T, n0,
        fixed["eps_e"], fixed["eps_B"], fixed["eps_T"], fixed["p"],
        fixed["mu_u"], fixed["mu_e"], BG0, alpha, fixed["k"], d_L, z,
        therm_el=therm_el, pl_el=pl_el,
    )
    return Lnu / (4.0 * np.pi * d_L ** 2) / C.Jy


def log_prob_dynamical(theta, epochs, fixed, therm_el, pl_el, bounds):
    lp = log_prior_box(theta, PARAM_LABELS_DYNAMICAL, bounds)
    a = theta[0]
    if not (0.5 < a < _a_upper_bound(fixed["p"], fixed["delta"])):
        return -np.inf
    if not np.isfinite(lp):
        return -np.inf

    try:
        chi2_total = 0.0
        for ep in epochs:
            Fnu_model = _fnu_dynamical(theta, ep["freq"], ep["T"], ep["z"],
                                        ep["d_L"], fixed, therm_el, pl_el)
            chi2_total += np.sum(((ep["flux"] - Fnu_model) / ep["eflux"]) ** 2)
        ll = -0.5 * chi2_total
    except Exception:
        return -np.inf

    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# =====================================================================
# MULTI_EPOCH MODE (joint fit via LOS_IHG_MULTI_EPOCH)
# =====================================================================
# Like 'dynamical' mode, R(t) is DERIVED from a shared deceleration law
# across ALL epochs of a source at once -- but LOS_IHG_MULTI_EPOCH evaluates
# every epoch's R(t) via one scipy.optimize.root_scalar solve each (through
# R0()/radius_implicit_calc()), not the ~8000-100000-per-epoch EATS
# integral Shell.py/L_ELOS_IHG needs.
#
# Calling convention is different from every other mode here: rather than
# evaluating one epoch's flux at a time, LOS_IHG_MULTI_EPOCH takes a SHARED
# frequency array and a FULL array of epoch times, returning a
# (n_epoch, n_freq) grid in one call -- required for correctness, not just
# convenience, since R_ref is computed once (referenced to the EARLIEST
# epoch when logR_dec=inf) and must be shared consistently across every
# other epoch's R(t) solve within that same likelihood evaluation.

def get_multi_epoch_data(source):
    """Gathers ALL epochs of a source for joint multi_epoch fitting, sorted
    by T ascending -- index 0 is the EARLIEST epoch (the reference epoch
    when logR_dec=inf). Builds a shared, deduplicated frequency array
    (nu_unique) across every epoch's own observed frequencies, plus an
    index mapping from each epoch's own freq array into nu_unique (exact
    match, since nu_unique is built directly from the real data)."""
    all_epochs = get_all_epochs(source)
    # drop any epoch with zero data (or a NaN T, e.g. from a fully-excluded
    # non-detection epoch) -- can't contribute to a joint fit and would
    # corrupt the R_ref/reference-epoch logic if included
    epochs = [ep for ep in all_epochs if len(ep["freq"]) > 0 and np.isfinite(ep["T"])]
    if not epochs:
        raise ValueError(f"No epochs with usable data found for source '{source}'")
    epochs = sorted(epochs, key=lambda ep: ep["T"])

    T_array = np.array([ep["T"] for ep in epochs])
    nu_unique = np.unique(np.concatenate([ep["freq"] for ep in epochs]))
    freq_idx_per_epoch = [np.searchsorted(nu_unique, ep["freq"]) for ep in epochs]

    return dict(epochs=epochs, T_array=T_array, nu_unique=nu_unique,
                freq_idx_per_epoch=freq_idx_per_epoch,
                z=epochs[0]["z"], d_L=epochs[0]["d_L"])


def _fnu_multi_epoch_grid(theta, labels, multi_data, fixed, logR_dec, pv_form,
                          therm_el, pl_el):
    """Evaluates LOS_IHG_MULTI_EPOCH once over multi_data['nu_unique'] and
    multi_data['T_array']. Returns the full (n_epoch, n_freq) flux grid in
    Jy (not luminosity), plus the params dict used and the per-epoch BG
    array LOS_IHG_MULTI_EPOCH also returns."""
    p = _build_theta_params(theta, labels, fixed)
    L_avg, BG = flux_variables.LOS_IHG_MULTI_EPOCH(
        multi_data["nu_unique"], logR_dec, p["s"], p["a"], p["delta"],
        multi_data["T_array"], p["n0"], p["eps_e"], p["eps_B"], p["eps_T"],
        p["p"], p["mu_u"], p["mu_e"], p["pv_ref"], p["k"], p["alpha"],
        multi_data["d_L"], multi_data["z"], pv_form,
        therm_el=therm_el, pl_el=pl_el,
    )
    Fnu_grid_Jy = L_avg / (4.0 * np.pi * multi_data["d_L"] ** 2) / C.Jy
    return Fnu_grid_Jy, BG, p


def log_prob_multi_epoch(theta, multi_data, fixed, therm_el, pl_el, bounds,
                          labels, logR_dec, pv_form):
    lp = log_prior_box(theta, labels, bounds)
    if not np.isfinite(lp):
        return -np.inf

    p = _build_theta_params(theta, labels, fixed)
    if not (0.5 < p["a"] < _a_upper_bound(p["p"], p["delta"])):
        return -np.inf
    if p["eps_e"] + p["eps_B"] + p["eps_T"] > 1.0:
        return -np.inf

    try:
        L_avg, BG = flux_variables.LOS_IHG_MULTI_EPOCH(
            multi_data["nu_unique"], logR_dec, p["s"], p["a"], p["delta"],
            multi_data["T_array"], p["n0"], p["eps_e"], p["eps_B"], p["eps_T"],
            p["p"], p["mu_u"], p["mu_e"], p["pv_ref"], p["k"], p["alpha"],
            multi_data["d_L"], multi_data["z"], pv_form,
            therm_el=therm_el, pl_el=pl_el,
        )
        chi2_total = 0.0
        for i, ep in enumerate(multi_data["epochs"]):
            idx = multi_data["freq_idx_per_epoch"][i]
            Fnu_model_Jy = L_avg[i, idx] / (4.0 * np.pi * multi_data["d_L"] ** 2) / C.Jy
            chi2_total += np.sum(((ep["flux"] - Fnu_model_Jy) / ep["eflux"]) ** 2)
        ll = -0.5 * chi2_total
    except Exception:
        return -np.inf

    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# =====================================================================
# WALKER INITIALIZATION
# =====================================================================

def sample_valid_initial_walkers(nwalkers, labels, bounds, log_prob_fn,
                                  log_prob_args, pool=None, max_batches=50,
                                  oversample=3, seed=None):
    ndim = len(labels)
    lb = np.array([bounds[l][0] for l in labels])
    ub = np.array([bounds[l][1] for l in labels])
    rng = np.random.default_rng(seed)

    accepted = []
    batch = 0
    while len(accepted) < nwalkers and batch < max_batches:
        batch += 1
        n_needed = nwalkers - len(accepted)
        trial = lb + rng.random((n_needed * oversample, ndim)) * (ub - lb)

        if pool is not None:
            lps = pool.starmap(log_prob_fn, [(t,) + log_prob_args for t in trial])
        else:
            lps = [log_prob_fn(t, *log_prob_args) for t in trial]

        for t, lp in zip(trial, lps):
            if np.isfinite(lp):
                accepted.append(t)
            if len(accepted) >= nwalkers:
                break
        print(f"    walker init batch {batch}: {len(accepted)}/{nwalkers} valid")

    if len(accepted) < nwalkers:
        raise RuntimeError(
            "Could not find enough valid initial walkers after "
            f"{max_batches} batches -- check priors / fixed params "
            "(eps_e+eps_B+eps_T, a range, etc.)."
        )
    return np.array(accepted[:nwalkers])


# =====================================================================
# SAMPLER DRIVER (HDF5-backed, resumable, optional early-stopping)
# =====================================================================

def run_sampler(tag, data_dir, ndim, labels, bounds, log_prob_fn, log_prob_args,
                 cfg):
    os.makedirs(data_dir, exist_ok=True)
    backend_path = os.path.join(data_dir, f"{tag}_chain.h5")
    backend = HDFBackend(backend_path)

    resuming = cfg["resume"] and os.path.exists(backend_path) and backend.iteration > 0
    if resuming:
        if backend.shape != (cfg["nwalkers"], ndim):
            raise RuntimeError(
                f"Cannot resume: existing backend has shape {backend.shape}, "
                f"but requested (nwalkers={cfg['nwalkers']}, ndim={ndim}). "
                f"Either match nwalkers or use a new --dir/RUN_TAG."
            )
        print(f"    resuming '{tag}' from existing chain "
              f"({backend.iteration} steps already stored)")
        p0 = None
    else:
        backend.reset(cfg["nwalkers"], ndim)
        p0 = None  # filled in below, inside the pool context

    with mp.get_context("spawn").Pool(processes=cfg["ncores"]) as pool:
        if not resuming:
            p0 = sample_valid_initial_walkers(
                cfg["nwalkers"], labels, bounds, log_prob_fn, log_prob_args,
                pool=pool, seed=cfg["seed"])

        sampler = emcee.EnsembleSampler(
            cfg["nwalkers"], ndim, log_prob_fn, args=log_prob_args,
            pool=pool, backend=backend)

        if cfg["check_convergence"]:
            old_tau = np.inf
            for _ in sampler.sample(p0, iterations=cfg["nsamples"], progress=True):
                if sampler.iteration % cfg["convergence_check_every"]:
                    continue
                tau = sampler.get_autocorr_time(tol=0)
                converged = np.all(tau * cfg["convergence_ntau"] < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < cfg["convergence_rtol"])
                if converged:
                    print(f"    converged early at iteration {sampler.iteration} "
                          f"(tau={tau})")
                    break
                old_tau = tau
        else:
            sampler.run_mcmc(p0, cfg["nsamples"], progress=True)

    return sampler


def get_flat_samples(sampler):
    try:
        tau = sampler.get_autocorr_time()
        discard = int(np.max(tau) * 3)
        thin = max(1, int(np.max(tau) / 2))
        print(f"    autocorr time: {tau} -> discard={discard}, thin={thin}")
    except emcee.autocorr.AutocorrError as e:
        print(f"    WARNING: autocorr time unreliable ({e}); "
              f"falling back to 20% burn-in, thin=1. Consider more steps.")
        discard = int(0.2 * sampler.iteration)
        thin = 1
    flat = sampler.get_chain(discard=discard, thin=thin, flat=True)
    flat_lp = sampler.get_log_prob(discard=discard, thin=thin, flat=True)
    return flat, flat_lp, discard, thin


# =====================================================================
# SAVING + PLOTTING
# =====================================================================

def save_chain_outputs(data_dir, plots_dir, tag, title_str, sampler, labels, T=None):
    flat, flat_lp, discard, thin = get_flat_samples(sampler)

    # post-burn-in, thinned samples -- one row per posterior draw, columns
    # are the free params (in the untransformed/raw theta space used by
    # the sampler, e.g. log10R not R) plus log_prob. These + the .h5 chain
    # are the "big" files -- kept under data/, gitignore that subtree.
    df = pd.DataFrame(flat, columns=labels)
    df["log_prob"] = flat_lp
    df.to_csv(os.path.join(data_dir, f"{tag}_flatMCMC.csv"), index=False)

    # compact one-row-per-run summary (median, 68% interval, MAP best) --
    # small, also lives in data/ since make_evolution_plot reads it from
    # there, but it's tiny (kept out of gitignore if you want to track it).
    summary = {"tag": tag}
    if T is not None:
        summary["T"] = T
    for j, lab in enumerate(labels):
        q16, q50, q84 = np.percentile(flat[:, j], [16, 50, 84])
        summary[f"{lab}_median"] = q50
        summary[f"{lab}_lo68"]   = q16
        summary[f"{lab}_hi68"]   = q84
        summary[f"{lab}_best"]   = flat[np.argmax(flat_lp), j]
    pd.DataFrame([summary]).to_csv(
        os.path.join(data_dir, f"{tag}_summary.csv"), index=False)

    print(f"    [{tag}] posterior summary:")
    for lab in labels:
        med = summary[f"{lab}_median"]
        print(f"      {lab}: {med:.4g}  (+{summary[f'{lab}_hi68']-med:.3g} "
              f"/ -{med-summary[f'{lab}_lo68']:.3g})  "
              f"best={summary[f'{lab}_best']:.4g}")

    # ---- trace plot: pretty math labels, single legend for the discard
    # boundary line (instead of stuffing it into the title) ----
    raw_chain = sampler.get_chain()
    fig, axes = plt.subplots(len(labels), 1, figsize=(9, 2.2 * len(labels)),
                              sharex=True)
    for j, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(raw_chain[:, :, j], alpha=0.4, lw=0.6, color="steelblue")
        ax.set_ylabel(pretty_label(labels[j]))
        ax.axvline(discard, color="crimson", ls="--", lw=1)
    axes[-1].set_xlabel("step")
    discard_handle = plt.Line2D([0], [0], color="crimson", ls="--", lw=1,
                                 label="discard boundary")
    fig.legend(handles=[discard_handle], loc="upper right")
    fig.suptitle(f"{title_str} -- trace")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"{tag}_trace.png"), dpi=130)
    plt.close(fig)

    # ---- corner plot: pretty title + math labels ----
    fig = corner.corner(flat, labels=[pretty_label(l) for l in labels],
                         show_titles=True, title_fmt=".3g",
                         quantiles=[0.16, 0.5, 0.84])
    fig.suptitle(title_str, y=1.02)
    fig.savefig(os.path.join(plots_dir, f"{tag}_corner.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    return flat, flat_lp, summary


def plot_sed_fitted_R(plots_dir, tag, title_str, epoch_data, flat, flat_lp, fixed,
                       labels, therm_el, pl_el, n_draws=60):
    # Axis limits (both grid AND display) come directly from the data's own
    # min/max, padded by SED_PAD_DEX so the fit shows a little beyond the
    # data on each side. nu_grid is generated over EXACTLY this same range,
    # so the plotted lines reach the plot edges with no dead margin.
    freq_data = epoch_data["freq"]
    flux_data = epoch_data["flux"]
    log_flo, log_fhi = np.log10(freq_data.min()), np.log10(freq_data.max())
    xlim = (10 ** (log_flo - SED_PAD_DEX), 10 ** (log_fhi + SED_PAD_DEX))
    nu_grid = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 400)

    log_ylo, log_yhi = np.log10(flux_data.min()), np.log10(flux_data.max())
    ylim = (10 ** (log_ylo - SED_PAD_DEX), 10 ** (log_yhi + SED_PAD_DEX))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(freq_data, flux_data, yerr=epoch_data["eflux"],
                fmt="o", ms=5, color="k", zorder=5, label="Data")

    # Blue curves = SEDs evaluated at `n_draws` random DRAWS FROM THE FULL
    # POST-BURN-IN POSTERIOR (not a chi2-thresholded ensemble like the old
    # least-squares script's "keep everything within Delta-chi2" approach).
    # Because MCMC naturally spends more steps in high-likelihood regions,
    # denser/more-overlapping blue curves already reflect higher posterior
    # density -- no extra chi2 filtering needed on top.
    idx = np.random.choice(len(flat), size=min(n_draws, len(flat)), replace=False)
    for i in idx:
        Fnu = _fnu_fitted_R(flat[i], labels, nu_grid, epoch_data["T"], epoch_data["z"],
                             epoch_data["d_L"], fixed, therm_el, pl_el)
        ax.plot(nu_grid, Fnu, color="steelblue", alpha=0.15, lw=1)

    best = flat[np.argmax(flat_lp)]
    Fnu_best = _fnu_fitted_R(best, labels, nu_grid, epoch_data["T"], epoch_data["z"],
                              epoch_data["d_L"], fixed, therm_el, pl_el)
    # NOTE: "Max Likelihood" and "MAP" are the same point here specifically
    # because the priors are flat/uniform within bounds (log_prior is a
    # constant wherever finite) -- if you ever switch to non-uniform
    # priors, these would no longer coincide and this label should change.
    ax.plot(nu_grid, Fnu_best, color="crimson", lw=2.5, label="Max Likelihood")

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\nu$ (Hz)"); ax.set_ylabel(r"$F_\nu$ (Jy)")
    ax.set_title(title_str)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"{tag}_sed.png"), dpi=130)
    plt.close(fig)


def plot_sed_dynamical(plots_dir, tag, title_str, epochs, flat, flat_lp, fixed,
                        therm_el, pl_el, n_draws=40):
    ncols = min(3, len(epochs))
    nrows = int(np.ceil(len(epochs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                              constrained_layout=True)
    ax_flat = np.atleast_1d(axes).reshape(-1)

    idx = np.random.choice(len(flat), size=min(n_draws, len(flat)), replace=False)
    best = flat[np.argmax(flat_lp)]

    for k, ep in enumerate(epochs):
        ax = ax_flat[k]
        freq_data = ep["freq"]
        flux_data = ep["flux"]
        log_flo, log_fhi = np.log10(freq_data.min()), np.log10(freq_data.max())
        xlim = (10 ** (log_flo - SED_PAD_DEX), 10 ** (log_fhi + SED_PAD_DEX))
        nu_grid = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 300)

        log_ylo, log_yhi = np.log10(flux_data.min()), np.log10(flux_data.max())
        ylim = (10 ** (log_ylo - SED_PAD_DEX), 10 ** (log_yhi + SED_PAD_DEX))

        ax.errorbar(freq_data, flux_data, yerr=ep["eflux"], fmt="o", ms=5,
                    color="k", zorder=5)
        # Blue curves here are also random draws from the joint posterior
        # (same note as plot_sed_fitted_R applies).
        for i in idx:
            Fnu = _fnu_dynamical(flat[i], nu_grid, ep["T"], ep["z"], ep["d_L"],
                                  fixed, therm_el, pl_el)
            ax.plot(nu_grid, Fnu, color="steelblue", alpha=0.15, lw=1)
        Fnu_best = _fnu_dynamical(best, nu_grid, ep["T"], ep["z"], ep["d_L"],
                                   fixed, therm_el, pl_el)
        ax.plot(nu_grid, Fnu_best, color="crimson", lw=2, label="Max Likelihood")

        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"epoch {ep['epoch']} | T={ep['T']:.1f}d")
        ax.set_xlabel(r"$\nu$ (Hz)"); ax.set_ylabel(r"$F_\nu$ (Jy)")

    for k in range(len(epochs), len(ax_flat)):
        ax_flat[k].set_visible(False)

    fig.suptitle(title_str)
    fig.savefig(os.path.join(plots_dir, f"{tag}_sed_allepochs.png"), dpi=130)
    plt.close(fig)


# =====================================================================
# POST-HOC AGGREGATE PLOTS (read already-saved per-epoch fitted_R
# summaries/data back off disk -- no resampling, no live sampler needed)
# =====================================================================

def _beta_from_BG(bg):
    """Convert proper velocity Gamma*beta -> beta. Monotonic increasing
    for bg >= 0, so applying it directly to already-computed percentiles
    (median/lo68/hi68/best) is exact -- no need to reload full samples."""
    bg = np.asarray(bg, dtype=float)
    return bg / np.sqrt(1.0 + bg ** 2)


def _load_fitted_R_summaries(outdir, run_tag, source):
    data_rundir = os.path.join(outdir, "data", run_tag)
    pattern = os.path.join(data_rundir, f"{source}_fittedR_ep*",
                            f"{source}_fittedR_ep*_summary.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No per-epoch summaries found matching {pattern}. "
            f"Run MODE='fitted_R' for each epoch of '{source}' first."
        )
    rows = [pd.read_csv(f).iloc[0] for f in files]
    df = pd.DataFrame(rows).sort_values("T").reset_index(drop=True)

    # Unit conversions for display (evolution + density-profile plots only;
    # corner/trace plots intentionally keep the raw sampler parameterization
    # unchanged). 10**x and BG->beta are both monotonic increasing, so
    # transforming the already-computed percentile columns directly is
    # exact -- no need to reload the full posterior.
    for stat in ["median", "lo68", "hi68", "best"]:
        df[f"R_{stat}"]    = 10.0 ** df[f"log10R_{stat}"]
        df[f"n0_{stat}"]   = 10.0 ** df[f"log10n0_{stat}"]
        df[f"beta_{stat}"] = _beta_from_BG(df[f"BG_{stat}"])
        # eps_B only present if it was a FREE param for this run (e.g. a
        # run4/5-style config) -- auto-detected rather than assumed.
        if f"log10eps_B_{stat}" in df.columns:
            df[f"eps_B_{stat}"] = 10.0 ** df[f"log10eps_B_{stat}"]

    return df


EVOLUTION_DISPLAY_PARAMS = ["a", "R", "beta", "n0"]
EVOLUTION_AXIS_INFO = {
    "a":     (r"$a$",                    False),
    "R":     (r"$R$ (cm)",               True),
    "beta":  (r"$\beta$",                False),
    "n0":    (r"$n_0$ (cm$^{-3}$)",      True),
    "eps_B": (r"$\epsilon_B$",           True),
}


def make_evolution_plot(outdir, run_tag, source):
    plots_rundir = os.path.join(outdir, "plots", run_tag)
    os.makedirs(plots_rundir, exist_ok=True)
    df = _load_fitted_R_summaries(outdir, run_tag, source)

    # base 4 always shown; eps_B only if it was actually free for this run
    display_params = list(EVOLUTION_DISPLAY_PARAMS)
    if "eps_B_median" in df.columns:
        display_params.append("eps_B")

    fig, axes = plt.subplots(len(display_params), 1,
                              figsize=(7, 2.5 * len(display_params)),
                              sharex=True)
    for j, (ax, key) in enumerate(zip(np.atleast_1d(axes), display_params)):
        med  = df[f"{key}_median"].values
        lo   = df[f"{key}_lo68"].values
        hi   = df[f"{key}_hi68"].values
        best = df[f"{key}_best"].values
        yerr = np.vstack([med - lo, hi - med])
        ax.errorbar(df["T"], med, yerr=yerr, fmt="o", capsize=4,
                    color="crimson", label="median +/- 68%")
        ax.plot(df["T"], best, "x--", color="steelblue", alpha=0.7,
                label="Max Likelihood")
        label, use_logy = EVOLUTION_AXIS_INFO[key]
        ax.set_ylabel(label)
        if use_logy:
            ax.set_yscale("log")
        if j == 0:
            ax.legend(fontsize=9)

    axes[-1].set_xlabel("T (observer-frame days)")
    name = SOURCE_DISPLAY_NAMES.get(source, source)
    fig.suptitle(f"{name} -- Parameter Evolution")
    fig.tight_layout()
    outpath = os.path.join(plots_rundir, f"{source}_param_evolution.png")
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"    saved -> {outpath}")

    make_density_profile_plot(df, plots_rundir, source)
    return df


def make_density_profile_plot(df, plots_rundir, source):
    """n0 (cm^-3) vs R (cm), points connected in time order (i.e. tracing
    the shock's actual trajectory through (R, n0) space as it expands) --
    median +/- 68% in red, Max Likelihood (no error bars) in faint blue,
    each with its own dashed connecting line. Also overlays m=-2/m=-3
    power-law reference slopes anchored on the first (earliest-epoch)
    Max Likelihood point."""
    fig, ax = plt.subplots(figsize=(7, 5))

    yerr_med = np.vstack([df["n0_median"] - df["n0_lo68"],
                          df["n0_hi68"] - df["n0_median"]])
    ax.errorbar(df["R_median"], df["n0_median"], yerr=yerr_med, fmt="o--",
                capsize=4, color="crimson", label="median +/- 68%")
    ax.plot(df["R_best"], df["n0_best"], "x--", color="steelblue", alpha=0.4,
            label="Max Likelihood")

    ax.set_xscale("log"); ax.set_yscale("log")
    # capture the data-driven view before adding reference lines, so the
    # power-law overlays span the full plot width without stretching the
    # axes themselves (their y-values can shoot far outside the data range
    # at the edges since they're steep slopes)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    R_anchor, n0_anchor = df["R_best"].iloc[0], df["n0_best"].iloc[0]
    R_line = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 200)
    ax.plot(R_line, n0_anchor * (R_line / R_anchor) ** -2,
            color="black", ls="-.", lw=1.2, label="m=-2")
    ax.plot(R_line, n0_anchor * (R_line / R_anchor) ** -3,
            color="black", ls=":", lw=1.2, label="m=-3")

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel(r"$R$ (cm)"); ax.set_ylabel(r"$n_0$ (cm$^{-3}$)")
    name = SOURCE_DISPLAY_NAMES.get(source, source)
    ax.set_title(f"{name} -- Density Profile")
    ax.legend()
    fig.tight_layout()
    outpath = os.path.join(plots_rundir, f"{source}_density_profile.png")
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"    saved -> {outpath}")


def make_sed_collage(cfg, fixed):
    """Grid of per-epoch SEDs (data + Max Likelihood fit only, no posterior
    draw lines), titled with the ACTUAL data time-span/mean of each epoch
    (not the EPOCH_GROUPS bin edges)."""
    source = cfg["source"]
    data_rundir = os.path.join(cfg["outdir"], "data", cfg["run_tag"])
    plots_rundir = os.path.join(cfg["outdir"], "plots", cfg["run_tag"])
    os.makedirs(plots_rundir, exist_ok=True)

    pattern = os.path.join(data_rundir, f"{source}_fittedR_ep*",
                            f"{source}_fittedR_ep*_summary.csv")
    files = sorted(glob.glob(pattern),
                   key=lambda f: int(re.search(r"ep(\d+)_summary", f).group(1)))
    if not files:
        raise FileNotFoundError(
            f"No per-epoch summaries found matching {pattern}. "
            f"Run MODE='fitted_R' for each epoch of '{source}' first."
        )

    n = len(files)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                              constrained_layout=True)
    ax_flat = np.atleast_1d(axes).reshape(-1)

    for k, f in enumerate(files):
        row = pd.read_csv(f).iloc[0]
        epoch_idx = int(re.search(r"ep(\d+)_summary", f).group(1))
        tag = f"{source}_fittedR_ep{epoch_idx}"
        ep = get_epoch_data(source, epoch_idx)
        # auto-discover which params were actually free for THIS epoch's
        # run, straight from its saved summary columns -- more robust than
        # trusting the current control panel's free-param list to match.
        best_cols = [c for c in row.index if c.endswith("_best")]
        free_labels = [c[:-len("_best")] for c in best_cols]
        theta_best = np.array([row[c] for c in best_cols])

        config_path = os.path.join(plots_rundir, tag, f"{tag}_config.txt")
        mismatches = check_fixed_params_match(config_path, fixed)
        if mismatches is None:
            print(f"    {tag}: no saved config found to verify against -- "
                  f"can't confirm FIXED_PARAMS matches what this epoch actually used.")
        elif mismatches:
            print(f"    {tag}: MISMATCH vs saved config -- " + "; ".join(mismatches))

        freq_data = ep["freq"]
        # Convert Jy -> specific luminosity (erg/s/Hz): L_nu = 4*pi*d_L^2 *
        # F_nu_cgs, and F_nu_cgs = F_nu_Jy * C.Jy (C.Jy = 1e-23 erg/s/cm^2/Hz).
        Lnu_conv = 4.0 * np.pi * ep["d_L"] ** 2 * C.Jy
        flux_data = ep["flux"] * Lnu_conv
        eflux_data = ep["eflux"] * Lnu_conv
        freq_nondet = ep["freq_nondet"]
        upper_limit_nondet = ep["upper_limit_nondet"] * Lnu_conv
        is_ul = ep["is_upper_limit"]

        # Split the FIT-participating points (ep["freq"]/ep["flux"]) by
        # whether they're a normal detection or a promoted non-detection
        # (still used in the fit, but displayed as an upper limit, not a
        # normal detection marker).
        freq_true_det = freq_data[~is_ul]
        flux_true_det = flux_data[~is_ul]
        eflux_true_det = eflux_data[~is_ul]
        freq_promoted = freq_data[is_ul]
        flux_promoted = flux_data[is_ul]

        all_freq = np.concatenate([freq_data, freq_nondet]) if len(freq_nondet) else freq_data
        all_flux = np.concatenate([flux_data, upper_limit_nondet]) if len(freq_nondet) else flux_data
        log_flo, log_fhi = np.log10(all_freq.min()), np.log10(all_freq.max())
        xlim = (10 ** (log_flo - SED_PAD_DEX), 10 ** (log_fhi + SED_PAD_DEX))
        nu_grid = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 400)
        log_ylo, log_yhi = np.log10(all_flux.min()), np.log10(all_flux.max())
        ylim = (10 ** (log_ylo - SED_PAD_DEX), 10 ** (log_yhi + SED_PAD_DEX))

        ax = ax_flat[k]
        ax.errorbar(freq_true_det, flux_true_det, yerr=eflux_true_det, fmt="o",
                    ms=5, color="k", zorder=5)
        # upper limits: dimmed, downward-pointing triangles, plotted at
        # SNR_THRESHOLD x their RMS noise (not the raw measured flux).
        # Includes BOTH true non-detections (excluded from the fit) and
        # any promoted non-detection (included in the fit, still shown
        # this way) -- same visual treatment for both.
        if len(freq_nondet):
            ax.scatter(freq_nondet, upper_limit_nondet, marker="v", s=45,
                       color="k", alpha=0.35, zorder=4)
        if len(freq_promoted):
            ax.scatter(freq_promoted, flux_promoted, marker="v", s=45,
                       color="k", alpha=0.35, zorder=4)
        Fnu_best = _fnu_fitted_R(theta_best, free_labels, nu_grid, ep["T"],
                                  ep["z"], ep["d_L"], fixed,
                                  cfg["therm_el"], cfg["pl_el"])
        Lnu_best = Fnu_best * Lnu_conv
        ax.plot(nu_grid, Lnu_best, color="crimson", lw=2.2)

        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{ep['t_rest_min']:.0f}-{ep['t_rest_max']:.0f} d "
                     r"($\langle t\rangle$=" + f"{ep['t_rest_mean']:.1f} d)")
        ax.set_xlabel(r"$\nu$ (Hz)")
        ax.set_ylabel(r"$L_\nu$ (ergs s$^{-1}$ Hz$^{-1}$)")

    for k in range(n, len(ax_flat)):
        ax_flat[k].set_visible(False)

    name = SOURCE_DISPLAY_NAMES.get(source, source)
    fig.suptitle(f"{name} -- SED Fits by Epoch")
    outpath = os.path.join(plots_rundir, f"{source}_sed_collage.png")
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved -> {outpath}")


# --- per-(source, epoch) reference power-law slopes to overlay on the
# SED overlay plot, anchored on the FIT CURVE (not the data point itself)
# at the frequency of the second-to-last (by frequency) SED point. Purely
# a visual comparison aid -- add more (source, epoch) entries as needed. ---
OVERLAY_REFERENCE_SLOPES = {
    "dbl": {3: [-2.5]},
}

# Fixed lower x-axis bound (rest-frame GHz) for the SED overlay plot --
# the fit curves are evaluated all the way down to this point too, so
# there's no dead margin between the lines and the axis edge.
OVERLAY_XLIM_LO_GHZ_REST = 10.0


def make_sed_overlay_plot(cfg, fixed):
    """Single-panel alternative to the SED collage: every epoch's data +
    Max Likelihood fit overlaid on one set of axes, colored by time
    (bluer = earlier/younger, redder = later/older). Each epoch's fit
    curve is labeled in the legend with its ACTUAL data time span (e.g.
    "30-34d"), not the EPOCH_GROUPS bin edges originally specified.

    Axes: y = flux density in uJy (log scale, no label by default -- set
    cfg["overlay_ylabel"]/--overlay_ylabel to add one), x = REST-FRAME
    frequency in GHz (log scale) -- note the model itself is still
    evaluated at the correct OBSERVED frequency/epoch internally; only the
    display x-values are converted to rest frame (nu_rest = nu_obs*(1+z)).

    Y-limits: fixed at [7.6, 3800] uJy.

    X-limits: set to exactly match the shared fit-curve frequency range
    (same range every epoch's curve is evaluated over), no extra padding
    beyond that.
    """
    source = cfg["source"]
    z = SOURCE_INFO[source]["z"]
    data_rundir = os.path.join(cfg["outdir"], "data", cfg["run_tag"])
    plots_rundir = os.path.join(cfg["outdir"], "plots", cfg["run_tag"])
    os.makedirs(plots_rundir, exist_ok=True)

    pattern = os.path.join(data_rundir, f"{source}_fittedR_ep*",
                            f"{source}_fittedR_ep*_summary.csv")
    files = sorted(glob.glob(pattern),
                   key=lambda f: int(re.search(r"ep(\d+)_summary", f).group(1)))
    if not files:
        raise FileNotFoundError(
            f"No per-epoch summaries found matching {pattern}. "
            f"Run MODE='fitted_R' for each epoch of '{source}' first."
        )

    # Gather everything per epoch first (need the full time range before
    # picking colors, and a single shared nu_grid -- in OBSERVED Hz, for
    # correct model evaluation -- to evaluate every epoch's fit curve on
    # for a consistent combined axis range).
    records = []
    all_freq_list = []
    for f in files:
        row = pd.read_csv(f).iloc[0]
        epoch_idx = int(re.search(r"ep(\d+)_summary", f).group(1))
        tag = f"{source}_fittedR_ep{epoch_idx}"
        ep = get_epoch_data(source, epoch_idx)

        best_cols = [c for c in row.index if c.endswith("_best")]
        free_labels = [c[:-len("_best")] for c in best_cols]
        theta_best = np.array([row[c] for c in best_cols])

        config_path = os.path.join(plots_rundir, tag, f"{tag}_config.txt")
        mismatches = check_fixed_params_match(config_path, fixed)
        if mismatches is None:
            print(f"    {tag}: no saved config found to verify against -- "
                  f"can't confirm FIXED_PARAMS matches what this epoch actually used.")
        elif mismatches:
            print(f"    {tag}: MISMATCH vs saved config -- " + "; ".join(mismatches))

        records.append(dict(epoch=epoch_idx, ep=ep, theta_best=theta_best,
                            free_labels=free_labels, t=ep["t_rest_mean"]))
        all_freq_list.append(ep["freq"])
        if len(ep["freq_nondet"]):
            all_freq_list.append(ep["freq_nondet"])

    times = np.array([r["t"] for r in records])
    cmap = plt.get_cmap("coolwarm")  # blue (low/young) -> red (high/old)
    norm = plt.Normalize(vmin=times.min(), vmax=times.max())

    # shared OBSERVED-frame nu_grid, same range for every epoch's curve.
    # Lower bound fixed at OVERLAY_XLIM_LO_GHZ_REST (rest-frame GHz),
    # converted down to the equivalent observed-frame Hz value so the fit
    # curves are actually evaluated (and thus drawn) all the way down to
    # that point -- no dead margin between the curve and the axis edge.
    all_freq = np.concatenate(all_freq_list)
    log_fhi = np.log10(all_freq.max())
    nu_grid_lo = (OVERLAY_XLIM_LO_GHZ_REST * 1e9) / (1 + z)
    nu_grid_hi = 10 ** (log_fhi + SED_PAD_DEX)
    nu_grid = np.logspace(np.log10(nu_grid_lo), np.log10(nu_grid_hi), 500)
    # rest-frame GHz version of that same grid, for display/x-limits only
    nu_grid_disp = nu_grid * (1 + z) / 1e9

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for r in records:
        ep = r["ep"]
        color = cmap(norm(r["t"]))

        is_ul = ep["is_upper_limit"]
        freq_true_det = ep["freq"][~is_ul]
        flux_true_det = ep["flux"][~is_ul] * 1e6      # Jy -> uJy
        eflux_true_det = ep["eflux"][~is_ul] * 1e6
        freq_promoted = ep["freq"][is_ul]
        flux_promoted = ep["flux"][is_ul] * 1e6
        freq_nondet = ep["freq_nondet"]
        upper_limit_nondet = ep["upper_limit_nondet"] * 1e6

        # display x-values: rest-frame GHz
        disp_true_det = freq_true_det * (1 + z) / 1e9
        disp_promoted = freq_promoted * (1 + z) / 1e9
        disp_nondet = freq_nondet * (1 + z) / 1e9

        ax.errorbar(disp_true_det, flux_true_det, yerr=eflux_true_det, fmt="o",
                    ms=5, color=color, zorder=5)
        if len(freq_promoted):
            ax.scatter(disp_promoted, flux_promoted, marker="v", s=45,
                       color=color, alpha=0.35, zorder=4)
        if len(freq_nondet):
            ax.scatter(disp_nondet, upper_limit_nondet, marker="v", s=45,
                       color=color, alpha=0.35, zorder=4)

        Fnu_best = _fnu_fitted_R(r["theta_best"], r["free_labels"], nu_grid,
                                  ep["T"], ep["z"], ep["d_L"], fixed,
                                  cfg["therm_el"], cfg["pl_el"])
        flux_best_uJy = Fnu_best * 1e6
        # Legend label = the ACTUAL data time span for this epoch (not the
        # EPOCH_GROUPS bin edges you originally specified), e.g. "30-34d".
        time_label = f" {ep['t_rest_min']:.0f}-{ep['t_rest_max']:.0f}d"
        ax.plot(nu_grid_disp, flux_best_uJy, color=color, lw=2.0, label=time_label)

        ref_slopes = OVERLAY_REFERENCE_SLOPES.get(source, {}).get(r["epoch"])
        if ref_slopes:
            # Anchor: x = second-to-last (by frequency) SED point; y = the
            # FIT CURVE evaluated at that x (not the data point's own flux
            # value) -- so the reference lines start exactly on the fit.
            freq_sorted = np.sort(ep["freq"])
            if len(freq_sorted) >= 2:
                anchor_x = freq_sorted[-2]
                anchor_y_uJy = _fnu_fitted_R(r["theta_best"], r["free_labels"],
                                             np.array([anchor_x]), ep["T"], ep["z"],
                                             ep["d_L"], fixed, cfg["therm_el"],
                                             cfg["pl_el"])[0] * 1e6
                nu_line = nu_grid[nu_grid >= anchor_x]
                nu_line_disp = nu_line * (1 + z) / 1e9
                for m in ref_slopes:
                    ref_curve = anchor_y_uJy * (nu_line / anchor_x) ** m
                    ax.plot(nu_line_disp, ref_curve, color=color, ls="--", lw=1.3,
                            alpha=0.85, label=rf"Power law ($\nu^{{{m}}}$)")

    ax.set_xscale("log"); ax.set_yscale("log")

    # x-limits: exactly the shared fit-curve range, nothing more
    ax.set_xlim(nu_grid_disp.min(), nu_grid_disp.max())

    # y-limits: fixed per your spec
    ax.set_ylim([7.6, 3800])

    ax.set_xlabel(r"$\nu_{\rm rest}$ (GHz)", fontsize=16)
    ylabel = cfg.get("overlay_ylabel")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.set_xticks([10, 50, 100], ["10", "50", "100"])

    # Every epoch's fit curve carries a time-span label, plus any
    # reference power-law lines -- legend() picks up all of them.
    ax.legend(loc="lower center", ncols=1, prop={"size": 10}, labelspacing=0.1,
              handletextpad=0.1, columnspacing=1)

    name = SOURCE_DISPLAY_NAMES.get(source, source)
    ax.text(0.97, 0.97, name, transform=ax.transAxes, ha="right", va="top",
            fontsize=14)

    outpath = os.path.join(plots_rundir, f"{source}_sed_overlay.png")
    # Explicit margins (not tight_layout/bbox_inches="tight") so every
    # save comes out at EXACTLY the (8, 4.5) canvas size regardless of
    # content, while still leaving enough room that labels/ticks don't
    # get clipped at these font sizes.
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.95)
    fig.savefig(outpath, dpi=450)
    plt.close(fig)
    print(f"    saved -> {outpath}")


def make_slope_collage(cfg):
    """Grid of per-epoch slope-vs-frequency: for N fit-participating SED
    data points (sorted by frequency), computes the N-1 discrete slopes
    between CONSECUTIVE points (not a smooth fit-curve derivative), each
    plotted at the geometric mean frequency of its bracketing pair. Pure
    data diagnostic -- doesn't need any fit to already exist, since it
    reads straight from get_epoch_data()."""
    source = cfg["source"]
    plots_rundir = os.path.join(cfg["outdir"], "plots", cfg["run_tag"])
    os.makedirs(plots_rundir, exist_ok=True)

    n_epochs = len(EPOCH_GROUPS[source])
    ncols = min(3, n_epochs)
    nrows = int(np.ceil(n_epochs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                              constrained_layout=True)
    ax_flat = np.atleast_1d(axes).reshape(-1)
    epoch_slopes = {}  # epoch_idx -> array of slopes, for the summary printed at the end

    for k in range(n_epochs):
        epoch_idx = k + 1
        ep = get_epoch_data(source, epoch_idx)
        ax = ax_flat[k]

        order = np.argsort(ep["freq"])
        freq_s = ep["freq"][order]
        flux_s = ep["flux"][order]

        if len(freq_s) < 2:
            ax.set_xscale("log")
            ax.text(0.5, 0.5, "< 2 points\n(no slope)", ha="center", va="center",
                    transform=ax.transAxes, color="0.5")
            epoch_slopes[epoch_idx] = np.array([])
        else:
            log_nu = np.log(freq_s)
            log_F = np.log(np.abs(flux_s))
            slopes = np.diff(log_F) / np.diff(log_nu)
            # representative x-position for each slope: geometric mean of
            # the two frequencies bracketing it
            nu_mid = np.sqrt(freq_s[:-1] * freq_s[1:])

            ax.plot(nu_mid, slopes, "o-", color="crimson", lw=1.5, ms=5)
            ax.axhline(0, color="0.75", lw=0.8, ls=":")
            ax.set_xscale("log")
            epoch_slopes[epoch_idx] = slopes

        ax.set_title(f"{ep['t_rest_min']:.0f}-{ep['t_rest_max']:.0f} d "
                     r"($\langle t\rangle$=" + f"{ep['t_rest_mean']:.1f} d)")
        ax.set_xlabel(r"$\nu$ (Hz)")
        ax.set_ylabel(r"slope $d\ln F_\nu/d\ln\nu$")

    for k in range(n_epochs, len(ax_flat)):
        ax_flat[k].set_visible(False)

    name = SOURCE_DISPLAY_NAMES.get(source, source)
    fig.suptitle(f"{name} -- Slope vs Frequency (from SED data points)")
    outpath = os.path.join(plots_rundir, f"{source}_slope_collage.png")
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved -> {outpath}")

    # Written to a file (not printed) -- lives alongside the collage png
    # in this run's plots folder, e.g. mcmc_output/plots/run1/
    slopes_path = os.path.join(plots_rundir, f"{source}_slopes.txt")
    lines = [f"{name} slopes by epoch", "=" * (len(name) + 16), ""]
    for epoch_idx, slopes in epoch_slopes.items():
        slopes_str = ", ".join(f"{s:.3f}" for s in slopes) if len(slopes) else "(< 2 points, no slope)"
        lines.append(f"epoch {epoch_idx}: {slopes_str}")
    with open(slopes_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"    saved -> {slopes_path}")


def make_highfreq_fit_collage(cfg):
    """Grid of per-epoch SEDs (all data points + non-detections for
    context) with a linear fit (in log-log space) overlaid as a dashed
    line, using ONLY the points above that epoch's HIGH_FREQ_THRESHOLD_HZ
    cutoff -- isolates the optically-thin spectral slope where it's
    cleanly sampled. A vertical line marks the cutoff itself. Legend on
    the fit line is just 'm=...' / 'p=...' (m = fitted slope,
    p = -2m+1 = the implied power-law electron index). Pure data
    diagnostic -- no fit/model needed, reads straight from get_epoch_data()."""
    source = cfg["source"]
    plots_rundir = os.path.join(cfg["outdir"], "plots", cfg["run_tag"])
    os.makedirs(plots_rundir, exist_ok=True)

    thresholds = HIGH_FREQ_THRESHOLD_HZ.get(source)
    if not thresholds:
        raise ValueError(f"No HIGH_FREQ_THRESHOLD_HZ entries defined for source '{source}'")

    epochs = sorted(thresholds.keys())
    n = len(epochs)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                              constrained_layout=True)
    ax_flat = np.atleast_1d(axes).reshape(-1)

    for k, epoch_idx in enumerate(epochs):
        ep = get_epoch_data(source, epoch_idx)
        ax = ax_flat[k]
        threshold = thresholds[epoch_idx]

        if len(ep["freq"]) == 0 and len(ep["freq_nondet"]) == 0:
            ax.set_xscale("log")
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="0.5")
            ax.set_title(f"epoch {epoch_idx}")
            continue

        Lnu_conv = 4.0 * np.pi * ep["d_L"] ** 2 * C.Jy
        freq_data = ep["freq"]
        flux_data = ep["flux"] * Lnu_conv
        eflux_data = ep["eflux"] * Lnu_conv
        is_ul = ep["is_upper_limit"]
        freq_nondet = ep["freq_nondet"]
        upper_limit_nondet = ep["upper_limit_nondet"] * Lnu_conv

        above = freq_data > threshold
        # points used in the fit: solid black. points not used: open/gray.
        ax.errorbar(freq_data[above & ~is_ul], flux_data[above & ~is_ul],
                    yerr=eflux_data[above & ~is_ul], fmt="o", ms=5,
                    color="k", zorder=5)
        ax.errorbar(freq_data[~above & ~is_ul], flux_data[~above & ~is_ul],
                    yerr=eflux_data[~above & ~is_ul], fmt="o", ms=5,
                    mfc="none", mec="k", ecolor="0.6", zorder=4)
        if len(freq_promoted := freq_data[is_ul]):
            ax.scatter(freq_promoted, flux_data[is_ul], marker="v", s=45,
                       color="k", alpha=0.35, zorder=4)
        if len(freq_nondet):
            ax.scatter(freq_nondet, upper_limit_nondet, marker="v", s=45,
                       color="k", alpha=0.35, zorder=4)

        all_freq = np.concatenate([freq_data, freq_nondet]) if len(freq_nondet) else freq_data
        all_flux = np.concatenate([flux_data, upper_limit_nondet]) if len(freq_nondet) else flux_data
        log_flo, log_fhi = np.log10(all_freq.min()), np.log10(all_freq.max())
        xlim = (10 ** (log_flo - SED_PAD_DEX), 10 ** (log_fhi + SED_PAD_DEX))
        log_ylo, log_yhi = np.log10(all_flux.min()), np.log10(all_flux.max())
        ylim = (10 ** (log_ylo - SED_PAD_DEX), 10 ** (log_yhi + SED_PAD_DEX))

        n_fit_pts = int(np.sum(above & ~is_ul))
        if n_fit_pts >= 2:
            fit_freq = freq_data[above & ~is_ul]
            fit_flux = flux_data[above & ~is_ul]
            log_nu = np.log10(fit_freq)
            log_F = np.log10(fit_flux)
            m, b = np.polyfit(log_nu, log_F, 1)
            p = -2.0 * m + 1.0

            nu_lo = 10 ** (np.log10(fit_freq.min()) - SED_PAD_DEX)
            nu_hi = 10 ** (np.log10(fit_freq.max()) + SED_PAD_DEX)
            nu_line = np.logspace(np.log10(nu_lo), np.log10(nu_hi), 50)
            F_line = 10 ** (m * np.log10(nu_line) + b)
            ax.plot(nu_line, F_line, color="crimson", ls="--", lw=2.0,
                    label=f"m={m:.2f}\np={p:.2f}")

            # Two-point secant slope: uses ONLY the first and last (by
            # frequency) points among the above-threshold subset, not a
            # full regression across all of them -- a simple derivative
            # estimate, plotted separately from the polyfit line above.
            fit_order = np.argsort(fit_freq)
            nu_1, nu_2 = fit_freq[fit_order[0]], fit_freq[fit_order[-1]]
            F_1, F_2 = fit_flux[fit_order[0]], fit_flux[fit_order[-1]]
            m_deriv = (np.log10(F_2) - np.log10(F_1)) / (np.log10(nu_2) - np.log10(nu_1))
            p_deriv = -2.0 * m_deriv + 1.0
            ax.plot([nu_1, nu_2], [F_1, F_2], color="steelblue", ls=":", lw=2.0,
                    marker="o", ms=5,
                    label=f"m_deriv={m_deriv:.2f}\np_deriv={p_deriv:.2f}")

            # Average of the individual point-to-point slopes within the
            # above-threshold subset (same method as make_slope_collage,
            # just restricted to this subset) -- a scalar summary, not a
            # single well-defined line, so it gets a legend-only proxy
            # handle (no line drawn for it).
            freq_sorted = fit_freq[fit_order]
            flux_sorted = fit_flux[fit_order]
            pointwise_slopes = np.diff(np.log10(flux_sorted)) / np.diff(np.log10(freq_sorted))
            m_avg = np.mean(pointwise_slopes)
            p_avg = -2.0 * m_avg + 1.0
            avg_handle = plt.Line2D([], [], color="none",
                                    label=f"m_avg={m_avg:.2f}\np_avg={p_avg:.2f}")

            handles, labels = ax.get_legend_handles_labels()
            handles.append(avg_handle)
            labels.append(avg_handle.get_label())
            ax.legend(handles=handles, labels=labels, fontsize=9, loc="best")
        else:
            ax.text(0.97, 0.97, "< 2 points\nabove cutoff", ha="right", va="top",
                    transform=ax.transAxes, fontsize=8, color="0.5")

        ax.axvline(threshold, color="0.5", lw=1.2, alpha=0.8)

        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{ep['t_rest_min']:.0f}-{ep['t_rest_max']:.0f} d "
                     r"($\langle t\rangle$=" + f"{ep['t_rest_mean']:.1f} d)")
        ax.set_xlabel(r"$\nu$ (Hz)")
        ax.set_ylabel(r"$L_\nu$ (ergs s$^{-1}$ Hz$^{-1}$)")

    for k in range(n, len(ax_flat)):
        ax_flat[k].set_visible(False)

    name = SOURCE_DISPLAY_NAMES.get(source, source)
    fig.suptitle(f"{name} -- High-Frequency Linear Fit by Epoch")
    outpath = os.path.join(plots_rundir, f"{source}_highfreq_fit_collage.png")
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved -> {outpath}")


# =====================================================================
# "TURNING KNOBS" SED SENSITIVITY COLLAGE
# =====================================================================
# 3x4 grid: top-left is the unchanged baseline SED; the other 11 panels
# each vary ONE parameter across KNOB_N_VALUES points (holding all others
# at the baseline/center), for the 4 fitted_R free params plus the 7
# fixed microphysical/geometric params (mu_u, mu_e excluded).

KNOB_ORDER = ["a", "log10R", "BG", "log10n0", "s", "delta", "eps_e", "eps_T",
              "p", "k", "eps_B"]


def _knob_bounds_and_logspace():
    bounds = dict(PRIORS_FITTED_R)
    bounds.update(KNOB_BOUNDS_FIXED)
    logspace = dict(KNOB_LOGSPACE_FREE)
    logspace.update(KNOB_LOGSPACE_FIXED)
    return bounds, logspace


def _knob_center(cfg):
    """Center/baseline values for all 11 dials, plus T/z/d_L to evaluate at.
    The 4 canonical fitted_R params (a, log10R, BG, log10n0) always come
    from here regardless of what a given run's --free_params happened to
    be; if --knob_from_summary points at a run where eps_B was ALSO free
    (e.g. a run4/5-style config), that fitted eps_B value is picked up too
    rather than silently falling back to FIXED_PARAMS's eps_B."""
    center = dict(FIXED_PARAMS)  # s, delta, eps_e, eps_T, p, mu_u, mu_e, k, eps_B
    z = SOURCE_INFO[cfg["knob_source"]]["z"]
    d_L = Planck18.luminosity_distance(z).cgs.value

    canonical = ["a", "log10R", "BG", "log10n0"]
    if cfg["knob_from_summary"]:
        df = pd.read_csv(cfg["knob_from_summary"])
        row = df.iloc[0]
        for lab in canonical:
            center[lab] = row[f"{lab}_best"]
        if "log10eps_B_best" in row:
            center["eps_B"] = 10.0 ** row["log10eps_B_best"]
        T = float(row["T"]) if "T" in row else cfg["knob_T"]
    else:
        for lab in canonical:
            center[lab] = float(np.mean(PRIORS_FITTED_R[lab]))
        T = cfg["knob_T"]

    return center, T, z, d_L


def _eval_knob_sed(nu, params, T, z, d_L, therm_el, pl_el):
    R = 10.0 ** params["log10R"]
    n0 = 10.0 ** params["log10n0"]
    Lnu = flux_variables.LOS_IHG_Fitted_R(
        nu, params["s"], params["a"], params["delta"], R, T, n0,
        params["eps_e"], params["eps_B"], params["eps_T"], params["p"],
        params["mu_u"], params["mu_e"], params["BG"], params["k"],
        params["alpha"], d_L, z,
        therm_el=therm_el, pl_el=pl_el,
    )
    return Lnu / (4.0 * np.pi * d_L ** 2) / C.Jy


def make_knob_plot(cfg, output_dir=None, filename_prefix=None):
    variant = cfg["knob_variant"]
    if variant == "pl_only":
        therm_el, pl_el = False, True
        variant_label, variant_suffix = "Power-Law Only", "plonly"
    elif variant == "therm_only":
        therm_el, pl_el = True, False
        variant_label, variant_suffix = "Thermal Only", "thermonly"
    else:
        therm_el, pl_el = cfg["therm_el"], cfg["pl_el"]
        variant_label, variant_suffix = "Combined", "combined"

    center, T, z, d_L = _knob_center(cfg)
    bounds, logspace = _knob_bounds_and_logspace()
    nu_grid = np.logspace(8, 13, 400)

    fig, axes = plt.subplots(3, 4, figsize=(20, 13))
    ax_flat = axes.reshape(-1)
    cmap = plt.get_cmap("viridis")

    # top-left: unchanged baseline
    base_curve = _eval_knob_sed(nu_grid, center, T, z, d_L, therm_el, pl_el)
    xlim, ylim = natural_xy_limits(nu_grid, [base_curve])
    ax0 = ax_flat[0]
    ax0.plot(nu_grid, base_curve, color="k", lw=2.5)
    ax0.set_xlim(xlim); ax0.set_ylim(ylim)
    ax0.set_xscale("log"); ax0.set_yscale("log")
    ax0.set_title("Baseline (unchanged)")
    ax0.set_xlabel(r"$\nu$ (Hz)"); ax0.set_ylabel(r"$F_\nu$ (Jy)")

    # Legend listing every dial's baseline value -- i.e. exactly what each
    # of the other 11 panels holds fixed while it varies its own parameter.
    baseline_handles = [
        plt.Line2D([], [], color="none",
                   label=f"{pretty_label(lab)} = {center[lab]:.3g}")
        for lab in KNOB_ORDER
    ]
    ax0.legend(handles=baseline_handles, fontsize=8, ncol=2,
               loc="upper right", framealpha=0.9,
               title="Baseline values", title_fontsize=9,
               handlelength=0, handletextpad=0)

    for k, lab in enumerate(KNOB_ORDER):
        ax = ax_flat[k + 1]
        lo, hi = bounds[lab]
        values = (np.logspace(np.log10(lo), np.log10(hi), cfg["knob_n_values"])
                  if logspace[lab] else
                  np.linspace(lo, hi, cfg["knob_n_values"]))

        curves = [base_curve]
        colors = cmap(np.linspace(0, 1, len(values)))
        for val, col in zip(values, colors):
            trial = dict(center)
            trial[lab] = val
            Fnu = _eval_knob_sed(nu_grid, trial, T, z, d_L, therm_el, pl_el)
            curves.append(Fnu)
            ax.plot(nu_grid, Fnu, color=col, lw=1.5,
                    label=f"{val:.3g}")

        ax.plot(nu_grid, base_curve, color="gray", lw=1.2, ls="--",
                alpha=0.7, label="baseline")

        xlim, ylim = natural_xy_limits(nu_grid, curves)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(pretty_label(lab))
        ax.set_xlabel(r"$\nu$ (Hz)"); ax.set_ylabel(r"$F_\nu$ (Jy)")
        ax.legend(fontsize=7, loc="best")

    name = SOURCE_DISPLAY_NAMES.get(cfg["knob_source"], cfg["knob_source"])
    center_note = (f"centered on {os.path.basename(cfg['knob_from_summary'])}"
                   if cfg["knob_from_summary"] else "centered on prior midpoints")
    fig.suptitle(f"{name} -- SED Sensitivity to Each Parameter -- {variant_label} "
                 f"(T={T:.1f}d, {center_note})", y=1.01)
    fig.tight_layout()

    plots_dir = output_dir or os.path.join(cfg["outdir"], "plots", cfg["run_tag"])
    os.makedirs(plots_dir, exist_ok=True)
    prefix = filename_prefix or f"knob_plot_{cfg['knob_source']}"
    outpath = os.path.join(plots_dir, f"{prefix}_{variant_suffix}.png")
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved -> {outpath}")


def make_all_knob_plots(cfg):
    """Runs the knob/dials plot for EVERY epoch that already has a saved
    fitted_R result for cfg['source'], each centered on that epoch's own
    best fit (like --knob_from_summary, just automatic), producing all 3
    variants (combined, pl_only, therm_only) per epoch. Output goes into
    a 'knobs/' subfolder inside that epoch's own plots directory, e.g.
    mcmc_output/plots/run1/wpp_fittedR_ep3/knobs/knob_plot_wpp_ep3_combined.png
    -- so everything for one epoch lives together."""
    source = cfg["source"]
    data_rundir = os.path.join(cfg["outdir"], "data", cfg["run_tag"])
    plots_rundir = os.path.join(cfg["outdir"], "plots", cfg["run_tag"])

    pattern = os.path.join(data_rundir, f"{source}_fittedR_ep*",
                            f"{source}_fittedR_ep*_summary.csv")
    files = sorted(glob.glob(pattern),
                   key=lambda f: int(re.search(r"ep(\d+)_summary", f).group(1)))
    if not files:
        raise FileNotFoundError(
            f"No per-epoch summaries found matching {pattern}. "
            f"Run MODE='fitted_R' for each epoch of '{source}' first."
        )

    for f in files:
        epoch_idx = int(re.search(r"ep(\d+)_summary", f).group(1))
        tag = f"{source}_fittedR_ep{epoch_idx}"
        knobs_dir = os.path.join(plots_rundir, tag, "knobs")

        epoch_cfg = dict(cfg)
        epoch_cfg["knob_source"] = source
        epoch_cfg["knob_from_summary"] = f

        print(f"  -- {tag} --")
        for variant in ["combined", "pl_only", "therm_only"]:
            epoch_cfg["knob_variant"] = variant
            make_knob_plot(epoch_cfg, output_dir=knobs_dir,
                           filename_prefix=f"knob_plot_{source}_ep{epoch_idx}")


# =====================================================================
# TIMING UTILITY
# =====================================================================

def time_single_eval(log_prob_fn, args, labels, bounds, n_reps=3):
    theta_mid = np.array([np.mean(bounds[lab]) for lab in labels])
    t0 = time.time()
    for _ in range(n_reps):
        log_prob_fn(theta_mid, *args)
    return (time.time() - t0) / n_reps


# =====================================================================
# RUNNERS
# =====================================================================

def run_fitted_R(source, epoch_idx, cfg, fixed):
    print(f"\n=== [fitted_R] {source} epoch {epoch_idx} ===")
    labels = cfg["free_params_fitted_R"]
    priors = cfg["priors_fitted_R"]
    ep = get_epoch_data(source, epoch_idx)
    weights = compute_post_peak_weights(ep["freq"], ep["flux"],
                                        cfg["chi2_weight_post_peak"],
                                        cfg["chi2_post_peak_weight"])
    if cfg["chi2_weight_post_peak"]:
        n_weighted = int(np.sum(weights > 1.0))
        print(f"    chi2 post-peak weighting ON: {n_weighted}/{len(weights)} "
              f"points above the empirical peak get x{cfg['chi2_post_peak_weight']} weight")
    args = (ep["freq"], ep["flux"], ep["eflux"], ep["T"], ep["z"], ep["d_L"],
             fixed, cfg["therm_el"], cfg["pl_el"], priors, labels, weights)

    if cfg["time_only"]:
        t_eval = time_single_eval(log_prob_fitted_R, args, labels, priors)
        est_total = t_eval * cfg["nwalkers"] * cfg["nsamples"]
        print(f"    single log_prob eval: {t_eval*1e3:.2f} ms")
        print(f"    est. total for nwalkers={cfg['nwalkers']}, "
              f"nsamples={cfg['nsamples']}: {est_total/3600:.2f} hr "
              f"(serial-equivalent; pool will help)")
        return

    tag = f"{source}_fittedR_ep{epoch_idx}"
    title_str = pretty_title(source, "fitted_R", epoch=epoch_idx)
    data_dir = os.path.join(cfg["outdir"], "data", cfg["run_tag"], tag)
    plots_dir = os.path.join(cfg["outdir"], "plots", cfg["run_tag"], tag)
    os.makedirs(plots_dir, exist_ok=True)

    save_run_config_txt(plots_dir, tag, fixed, priors,
                        cfg["therm_el"], cfg["pl_el"], free_labels=labels,
                        chi2_weight_post_peak=cfg["chi2_weight_post_peak"],
                        chi2_post_peak_weight=cfg["chi2_post_peak_weight"])

    sampler = run_sampler(tag, data_dir, len(labels), labels, priors,
                           log_prob_fitted_R, args, cfg)

    flat, flat_lp, _ = save_chain_outputs(data_dir, plots_dir, tag, title_str,
                                          sampler, labels, T=ep["T"])
    plot_sed_fitted_R(plots_dir, tag, title_str, ep, flat, flat_lp, fixed,
                       labels, cfg["therm_el"], cfg["pl_el"])


def run_dynamical(source, cfg, fixed):
    print(f"\n=== [dynamical] {source} (joint, all epochs) ===")
    epochs = get_all_epochs(source)
    args = (epochs, fixed, cfg["therm_el"], cfg["pl_el"], PRIORS_DYNAMICAL)

    if cfg["time_only"]:
        t_eval = time_single_eval(log_prob_dynamical, args,
                                   PARAM_LABELS_DYNAMICAL, PRIORS_DYNAMICAL)
        est_total = t_eval * cfg["nwalkers"] * cfg["nsamples"]
        print(f"    single log_prob eval ({len(epochs)} epochs): "
              f"{t_eval*1e3:.1f} ms")
        print(f"    est. total for nwalkers={cfg['nwalkers']}, "
              f"nsamples={cfg['nsamples']}: {est_total/3600:.2f} hr "
              f"(serial-equivalent; pool will help, but this mode is "
              f"inherently much slower than fitted_R)")
        return

    tag = f"{source}_dynamical_joint"
    title_str = pretty_title(source, "dynamical")
    data_dir = os.path.join(cfg["outdir"], "data", cfg["run_tag"], tag)
    plots_dir = os.path.join(cfg["outdir"], "plots", cfg["run_tag"], tag)
    os.makedirs(plots_dir, exist_ok=True)

    save_run_config_txt(plots_dir, tag, fixed, PRIORS_DYNAMICAL,
                        cfg["therm_el"], cfg["pl_el"])

    sampler = run_sampler(tag, data_dir, len(PARAM_LABELS_DYNAMICAL),
                           PARAM_LABELS_DYNAMICAL, PRIORS_DYNAMICAL,
                           log_prob_dynamical, args, cfg)

    flat, flat_lp, _ = save_chain_outputs(data_dir, plots_dir, tag, title_str,
                                          sampler, PARAM_LABELS_DYNAMICAL)
    plot_sed_dynamical(plots_dir, tag, title_str, epochs, flat, flat_lp, fixed,
                        cfg["therm_el"], cfg["pl_el"])


def plot_sed_multi_epoch(plots_dir, tag, title_str, multi_data, fixed, labels,
                         logR_dec, pv_form, flat, flat_lp, therm_el, pl_el,
                         n_draws=40):
    epochs = multi_data["epochs"]
    ncols = min(3, len(epochs))
    nrows = int(np.ceil(len(epochs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                              constrained_layout=True)
    ax_flat = np.atleast_1d(axes).reshape(-1)

    # Shared smooth nu_grid spanning the union of all epochs' frequencies --
    # a NEW multi_data dict reusing the same T_array (for R_ref
    # self-consistency) but this smooth grid instead of nu_unique.
    all_freq = np.concatenate([ep["freq"] for ep in epochs])
    log_flo, log_fhi = np.log10(all_freq.min()), np.log10(all_freq.max())
    nu_grid = np.logspace(log_flo - SED_PAD_DEX, log_fhi + SED_PAD_DEX, 300)
    plot_multi_data = dict(multi_data)
    plot_multi_data["nu_unique"] = nu_grid

    idx = np.random.choice(len(flat), size=min(n_draws, len(flat)), replace=False)
    best = flat[np.argmax(flat_lp)]

    Fnu_best_grid, _, _ = _fnu_multi_epoch_grid(best, labels, plot_multi_data,
                                                fixed, logR_dec, pv_form,
                                                therm_el, pl_el)
    draw_grids = []
    for i in idx:
        g, _, _ = _fnu_multi_epoch_grid(flat[i], labels, plot_multi_data,
                                        fixed, logR_dec, pv_form, therm_el, pl_el)
        draw_grids.append(g)

    for k, ep in enumerate(epochs):
        ax = ax_flat[k]
        freq_data, flux_data = ep["freq"], ep["flux"]
        log_flo_e, log_fhi_e = np.log10(freq_data.min()), np.log10(freq_data.max())
        xlim = (10 ** (log_flo_e - SED_PAD_DEX), 10 ** (log_fhi_e + SED_PAD_DEX))
        log_ylo, log_yhi = np.log10(flux_data.min()), np.log10(flux_data.max())
        ylim = (10 ** (log_ylo - SED_PAD_DEX), 10 ** (log_yhi + SED_PAD_DEX))

        ax.errorbar(freq_data, flux_data, yerr=ep["eflux"], fmt="o", ms=5,
                    color="k", zorder=5)
        # Blue curves = random draws from the joint posterior (same note
        # as plot_sed_fitted_R/plot_sed_dynamical applies).
        for g in draw_grids:
            ax.plot(nu_grid, g[k], color="steelblue", alpha=0.15, lw=1)
        ax.plot(nu_grid, Fnu_best_grid[k], color="crimson", lw=2,
                label="Max Likelihood")

        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"epoch {ep['epoch']} | T={ep['T']:.1f}d")
        ax.set_xlabel(r"$\nu$ (Hz)"); ax.set_ylabel(r"$F_\nu$ (Jy)")

    for k in range(len(epochs), len(ax_flat)):
        ax_flat[k].set_visible(False)

    fig.suptitle(title_str)
    outpath = os.path.join(plots_dir, f"{tag}_sed_allepochs.png")
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"    saved -> {outpath}")


def run_multi_epoch(source, cfg, fixed):
    print(f"\n=== [multi_epoch] {source} (joint, all epochs) ===")
    multi_data = get_multi_epoch_data(source)
    labels = cfg["free_params_multi_epoch"]
    priors = cfg["priors_multi_epoch"]
    logR_dec = cfg["multi_logR_dec"]
    pv_form = cfg["multi_pv_form"]
    args = (multi_data, fixed, cfg["therm_el"], cfg["pl_el"], priors, labels,
             logR_dec, pv_form)

    if cfg["time_only"]:
        t_eval = time_single_eval(log_prob_multi_epoch, args, labels, priors)
        est_total = t_eval * cfg["nwalkers"] * cfg["nsamples"]
        print(f"    single log_prob eval ({len(multi_data['epochs'])} epochs, "
              f"{len(multi_data['nu_unique'])} unique freqs): {t_eval*1e3:.1f} ms")
        print(f"    est. total for nwalkers={cfg['nwalkers']}, "
              f"nsamples={cfg['nsamples']}: {est_total/3600:.2f} hr "
              f"(serial-equivalent; pool will help)")
        return

    tag = f"{source}_multi_epoch_joint"
    title_str = pretty_title(source, "multi_epoch")
    data_dir = os.path.join(cfg["outdir"], "data", cfg["run_tag"], tag)
    plots_dir = os.path.join(cfg["outdir"], "plots", cfg["run_tag"], tag)
    os.makedirs(plots_dir, exist_ok=True)

    save_run_config_txt(plots_dir, tag, fixed, priors, cfg["therm_el"],
                        cfg["pl_el"], free_labels=labels)
    with open(os.path.join(plots_dir, f"{tag}_config.txt"), "a") as fh:
        fh.write(f"\nmulti_epoch logR_dec = {logR_dec}\n")
        fh.write(f"multi_epoch pv_form  = {pv_form}\n")
        fh.write(f"multi_epoch n_epochs = {len(multi_data['epochs'])}\n")

    sampler = run_sampler(tag, data_dir, len(labels), labels, priors,
                           log_prob_multi_epoch, args, cfg)

    flat, flat_lp, _ = save_chain_outputs(data_dir, plots_dir, tag, title_str,
                                          sampler, labels)
    plot_sed_multi_epoch(plots_dir, tag, title_str, multi_data, fixed, labels,
                         logR_dec, pv_form, flat, flat_lp,
                         cfg["therm_el"], cfg["pl_el"])


# =====================================================================
# REPLOT (regenerate plots from an ALREADY-SAVED chain, no resampling)
# =====================================================================
# The .h5 backend is the full, immutable record of what was sampled --
# HDFBackend already implements get_chain()/get_log_prob()/get_autocorr_time()
# /`.iteration`, i.e. everything save_chain_outputs() needs, so it can be
# handed to save_chain_outputs() in place of a live `sampler` with no
# changes to that function.
#
# CAVEAT: fixed microphysical params aren't stored in the chain itself (only
# in *_config.txt, which only exists for runs made after that feature was
# added). Replotting uses cfg["outdir"]'s CURRENT FIXED_PARAMS to draw the
# SED curves -- if you've changed the control panel since the original run,
# double check against that run's *_config.txt (if present) before trusting
# the regenerated SED panel; the corner/trace plots are unaffected either
# way since those come straight from the stored chain.

def replot_fitted_R(source, epoch_idx, cfg, fixed):
    tag = f"{source}_fittedR_ep{epoch_idx}"
    title_str = pretty_title(source, "fitted_R", epoch=epoch_idx)
    data_dir = os.path.join(cfg["outdir"], "data", cfg["run_tag"], tag)
    plots_dir = os.path.join(cfg["outdir"], "plots", cfg["run_tag"], tag)
    backend_path = os.path.join(data_dir, f"{tag}_chain.h5")
    summary_path = os.path.join(data_dir, f"{tag}_summary.csv")

    if not os.path.exists(backend_path):
        print(f"    SKIP {tag}: no chain found at {backend_path}")
        return
    if not os.path.exists(summary_path):
        print(f"    SKIP {tag}: no summary.csv found at {summary_path} "
              f"(needed to recover which params were free for this run)")
        return
    os.makedirs(plots_dir, exist_ok=True)

    # Recover the ACTUAL free-param labels used for this run from its own
    # saved summary -- robust regardless of what the CURRENT control panel's
    # FREE_PARAMS_FITTED_R happens to be (avoids ndim/label mismatches when
    # replotting a run made with a different free-parameter configuration,
    # e.g. one where eps_B was free and the current config doesn't have it).
    summary_row = pd.read_csv(summary_path).iloc[0]
    labels = [c[:-len("_best")] for c in summary_row.index if c.endswith("_best")]

    old_config = os.path.join(plots_dir, f"{tag}_config.txt")
    mismatches = check_fixed_params_match(old_config, fixed)
    if mismatches is None:
        print(f"    {tag}: no saved config found to verify against -- using "
              f"CURRENT FIXED_PARAMS for whatever wasn't free (free params "
              f"recovered from this run's own summary.csv: {labels})")
        write_config = True
    elif mismatches:
        print(f"    {tag}: MISMATCH vs saved config -- " + "; ".join(mismatches))
        print(f"    NOT overwriting the saved config.txt (it's still correct); "
              f"the SED panel below uses your CURRENT mismatched values -- "
              f"fix the override and rerun to get a correct SED.")
        write_config = False
    else:
        write_config = True

    backend = HDFBackend(backend_path)
    ep = get_epoch_data(source, epoch_idx)

    if write_config:
        save_run_config_txt(plots_dir, tag, fixed, cfg["priors_fitted_R"],
                            cfg["therm_el"], cfg["pl_el"], free_labels=labels)
    flat, flat_lp, _ = save_chain_outputs(data_dir, plots_dir, tag, title_str,
                                          backend, labels, T=ep["T"])
    plot_sed_fitted_R(plots_dir, tag, title_str, ep, flat, flat_lp, fixed,
                       labels, cfg["therm_el"], cfg["pl_el"])
    print(f"    replotted {tag}")


def replot_dynamical(source, cfg, fixed):
    tag = f"{source}_dynamical_joint"
    title_str = pretty_title(source, "dynamical")
    data_dir = os.path.join(cfg["outdir"], "data", cfg["run_tag"], tag)
    plots_dir = os.path.join(cfg["outdir"], "plots", cfg["run_tag"], tag)
    backend_path = os.path.join(data_dir, f"{tag}_chain.h5")

    if not os.path.exists(backend_path):
        print(f"    SKIP {tag}: no chain found at {backend_path}")
        return
    os.makedirs(plots_dir, exist_ok=True)

    old_config = os.path.join(plots_dir, f"{tag}_config.txt")
    mismatches = check_fixed_params_match(old_config, fixed)
    if mismatches is None:
        print(f"    {tag}: no saved config found to verify against -- "
              f"using CURRENT FIXED_PARAMS/priors to draw the SEDs.")
        write_config = True
    elif mismatches:
        print(f"    {tag}: MISMATCH vs saved config -- " + "; ".join(mismatches))
        print(f"    NOT overwriting the saved config.txt (it's still correct); "
              f"the SED panels below use your CURRENT mismatched values -- "
              f"fix the override and rerun to get correct SEDs.")
        write_config = False
    else:
        write_config = True

    backend = HDFBackend(backend_path)
    epochs = get_all_epochs(source)

    if write_config:
        save_run_config_txt(plots_dir, tag, fixed, PRIORS_DYNAMICAL,
                            cfg["therm_el"], cfg["pl_el"])
    flat, flat_lp, _ = save_chain_outputs(data_dir, plots_dir, tag, title_str,
                                          backend, PARAM_LABELS_DYNAMICAL)
    plot_sed_dynamical(plots_dir, tag, title_str, epochs, flat, flat_lp, fixed,
                        cfg["therm_el"], cfg["pl_el"])
    print(f"    replotted {tag}")


# =====================================================================
# CLI (all overrides are optional; unset ones fall back to CONTROL PANEL)
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=None, choices=["wpp", "dbl"])
    p.add_argument("--mode", default=None, choices=["fitted_R", "dynamical", "multi_epoch"])
    p.add_argument("--epoch", default=None,
                   help="1-indexed epoch, or 'all' (fitted_R mode only)")
    p.add_argument("--nwalkers", type=int, default=None)
    p.add_argument("--nsamples", type=float, default=None,
                   help="steps per walker; accepts sci notation e.g. 1e7")
    p.add_argument("--ncores", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dir", dest="run_tag", default=None,
                   help="output subdirectory label (like old $DIR)")
    p.add_argument("--outdir", default=None)
    p.add_argument("--resume", dest="resume", action="store_true", default=None)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--time_only", action="store_true", default=None)
    p.add_argument("--make_evolution_plot", action="store_true", default=None)
    p.add_argument("--make_sed_collage", action="store_true", default=None,
                   help="grid of per-epoch SEDs (data + Max Likelihood only) "
                        "for a source; use with --source/--dir")
    p.add_argument("--make_sed_overlay", action="store_true", default=None,
                   help="single-panel alternative to the SED collage: every "
                        "epoch overlaid, colored by time, each labeled in the "
                        "legend with its actual data time span; use with "
                        "--source/--dir")
    p.add_argument("--overlay_ylabel", default=None,
                   help="y-axis label for the SED overlay plot (default: none)")
    p.add_argument("--make_slope_collage", action="store_true", default=None,
                   help="grid of per-epoch slope-vs-frequency (discrete slopes "
                        "between consecutive SED data points); use with "
                        "--source/--dir, no fit required")
    p.add_argument("--make_highfreq_collage", action="store_true", default=None,
                   help="grid of per-epoch SEDs with a linear fit (log-log) "
                        "overlaid using only points above that epoch's "
                        "HIGH_FREQ_THRESHOLD_HZ cutoff; use with --source/--dir, "
                        "no MCMC fit required")
    p.add_argument("--replot", action="store_true", default=None,
                   help="regenerate plots from an existing saved chain "
                        "(no resampling); use with --source/--mode/--epoch/--dir "
                        "same as the original run")
    p.add_argument("--make_knob_plot", action="store_true", default=None)
    p.add_argument("--make_all_knob_plots", action="store_true", default=None,
                   help="runs all 3 knob-plot variants (combined/pl_only/"
                        "therm_only) for EVERY epoch fit of --source, each "
                        "centered on that epoch's own best fit, saved into "
                        "that epoch's own plots/.../knobs/ subfolder")
    p.add_argument("--knob_source", default=None, choices=["wpp", "dbl"])
    p.add_argument("--knob_T", type=float, default=None)
    p.add_argument("--knob_from_summary", default=None,
                   help="path to a *_summary.csv to center the free-param "
                        "dials on a real epoch's MAP fit (also uses its T)")
    p.add_argument("--knob_n_values", type=int, default=None)
    p.add_argument("--knob_variant", default=None,
                   choices=["combined", "pl_only", "therm_only"])
    p.add_argument("--therm_el", dest="therm_el", action="store_true", default=None)
    p.add_argument("--no-therm_el", dest="therm_el", action="store_false")
    p.add_argument("--pl_el", dest="pl_el", action="store_true", default=None)
    p.add_argument("--no-pl_el", dest="pl_el", action="store_false")

    # chi2 weighting for SED points past an epoch's empirical peak (fit
    # only -- never touches stored eflux/error bars/plots elsewhere)
    p.add_argument("--chi2_weight_post_peak", dest="chi2_weight_post_peak",
                   action="store_true", default=None,
                   help="upweight chi2 contribution of points above each "
                        "epoch's own empirical peak flux frequency")
    p.add_argument("--no-chi2_weight_post_peak", dest="chi2_weight_post_peak",
                   action="store_false")
    p.add_argument("--chi2_post_peak_weight", type=float, default=None,
                   help="chi2 multiplier for post-peak points when the above "
                        "is on (default: 3.0)")

    # fixed (non-fit) microphysical param overrides -- lets you launch
    # different configs (run2, run3, ...) without editing FIXED_PARAMS
    p.add_argument("--eps_B", type=float, default=None)
    p.add_argument("--eps_e", type=float, default=None)
    p.add_argument("--eps_T", type=float, default=None)
    p.add_argument("--p_index", type=float, default=None, dest="p_index",
                   help="power-law electron index p (renamed from --p to "
                        "avoid clashing with argparse's own conventions)")
    p.add_argument("--k", type=float, default=None)
    p.add_argument("--s", type=float, default=None)
    p.add_argument("--delta", type=float, default=None)

    # which params are free in fitted_R mode (default: FREE_PARAMS_FITTED_R)
    p.add_argument("--free_params", default=None,
                   help="comma-separated free-param list, e.g. "
                        "'a,log10R,BG,log10n0,log10eps_B' to fit eps_B too")
    p.add_argument("--log10eps_B_bounds", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="override the log10eps_B prior bounds (only used "
                        "if log10eps_B is in --free_params)")
    p.add_argument("--log10eps_ratio_bounds", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="override the log10eps_ratio (=eps_e/eps_T) prior "
                        "bounds (only used if log10eps_ratio is in --free_params)")

    # multi_epoch mode settings
    p.add_argument("--free_params_multi", default=None,
                   help="comma-separated free-param list for multi_epoch "
                        "mode, e.g. 'a,log10n0,pv_ref,alpha'")
    p.add_argument("--multi_logR_dec", type=float, default=None,
                   help="reference radius log10(cm) for multi_epoch mode; "
                        "pass 'inf' to auto-derive it from pv_ref/alpha/the "
                        "earliest epoch (default)")
    p.add_argument("--multi_pv_form", default=None, choices=["pl", "decel"],
                   help="deceleration functional form for multi_epoch mode")
    return p.parse_args()


def _resolve(cli_val, panel_val):
    return panel_val if cli_val is None else cli_val


def build_config():
    cli = parse_args()

    fixed = dict(FIXED_PARAMS)
    for key, cli_val in [("eps_B", cli.eps_B), ("eps_e", cli.eps_e),
                          ("eps_T", cli.eps_T), ("p", cli.p_index),
                          ("k", cli.k), ("s", cli.s), ("delta", cli.delta)]:
        if cli_val is not None:
            fixed[key] = cli_val

    free_params_fitted_R = (cli.free_params.split(",") if cli.free_params
                             else list(FREE_PARAMS_FITTED_R))

    priors_fitted_R = dict(PRIORS_FITTED_R)
    if cli.log10eps_B_bounds is not None:
        priors_fitted_R["log10eps_B"] = tuple(cli.log10eps_B_bounds)
    if cli.log10eps_ratio_bounds is not None:
        priors_fitted_R["log10eps_ratio"] = tuple(cli.log10eps_ratio_bounds)

    free_params_multi_epoch = (cli.free_params_multi.split(",") if cli.free_params_multi
                                else list(FREE_PARAMS_MULTI_EPOCH))
    priors_multi_epoch = dict(PRIORS_MULTI_EPOCH)

    cfg = dict(
        mode        = _resolve(cli.mode, MODE),
        source      = _resolve(cli.source, SOURCE),
        epoch       = _resolve(cli.epoch, EPOCH),
        nwalkers    = _resolve(cli.nwalkers, NWALKERS),
        nsamples    = int(_resolve(cli.nsamples, NSAMPLES)),
        ncores      = _resolve(cli.ncores, NCORES),
        seed        = _resolve(cli.seed, SEED),
        run_tag     = _resolve(cli.run_tag, RUN_TAG),
        outdir      = _resolve(cli.outdir, OUTDIR),
        resume      = _resolve(cli.resume, RESUME),
        time_only   = _resolve(cli.time_only, TIME_ONLY),
        make_evolution_plot = _resolve(cli.make_evolution_plot, MAKE_EVOLUTION_PLOT),
        make_sed_collage = _resolve(cli.make_sed_collage, MAKE_SED_COLLAGE),
        make_sed_overlay = _resolve(cli.make_sed_overlay, MAKE_SED_OVERLAY),
        overlay_ylabel = _resolve(cli.overlay_ylabel, OVERLAY_YLABEL),
        make_slope_collage = _resolve(cli.make_slope_collage, MAKE_SLOPE_COLLAGE),
        make_highfreq_collage = _resolve(cli.make_highfreq_collage, MAKE_HIGHFREQ_COLLAGE),
        replot      = _resolve(cli.replot, REPLOT),
        make_knob_plot = _resolve(cli.make_knob_plot, MAKE_KNOB_PLOT),
        make_all_knob_plots = _resolve(cli.make_all_knob_plots, MAKE_ALL_KNOB_PLOTS),
        knob_source = _resolve(cli.knob_source, KNOB_SOURCE),
        knob_T = _resolve(cli.knob_T, KNOB_T),
        knob_from_summary = _resolve(cli.knob_from_summary, KNOB_FROM_SUMMARY),
        knob_n_values = _resolve(cli.knob_n_values, KNOB_N_VALUES),
        knob_variant = _resolve(cli.knob_variant, KNOB_VARIANT),
        therm_el    = _resolve(cli.therm_el, THERM_EL),
        pl_el       = _resolve(cli.pl_el, PL_EL),
        chi2_weight_post_peak = _resolve(cli.chi2_weight_post_peak, CHI2_WEIGHT_POST_PEAK),
        chi2_post_peak_weight = _resolve(cli.chi2_post_peak_weight, CHI2_POST_PEAK_WEIGHT),
        check_convergence       = CHECK_CONVERGENCE,
        convergence_check_every = CONVERGENCE_CHECK_EVERY,
        convergence_ntau        = CONVERGENCE_NTAU,
        convergence_rtol        = CONVERGENCE_RTOL,
        fixed                 = fixed,
        free_params_fitted_R  = free_params_fitted_R,
        priors_fitted_R       = priors_fitted_R,
        free_params_multi_epoch = free_params_multi_epoch,
        priors_multi_epoch      = priors_multi_epoch,
        multi_logR_dec = _resolve(cli.multi_logR_dec, MULTI_EPOCH_LOGR_DEC),
        multi_pv_form  = _resolve(cli.multi_pv_form, MULTI_EPOCH_PV_FORM),
    )
    return cfg


def main():
    cfg = build_config()
    fixed = cfg["fixed"]
    check_physicality(fixed, cfg["free_params_fitted_R"])

    if cfg["make_evolution_plot"]:
        make_evolution_plot(cfg["outdir"], cfg["run_tag"], cfg["source"])
        return

    if cfg["make_sed_collage"]:
        make_sed_collage(cfg, fixed)
        return

    if cfg["make_sed_overlay"]:
        make_sed_overlay_plot(cfg, fixed)
        return

    if cfg["make_slope_collage"]:
        make_slope_collage(cfg)
        return

    if cfg["make_highfreq_collage"]:
        make_highfreq_fit_collage(cfg)
        return

    if cfg["make_knob_plot"]:
        make_knob_plot(cfg)
        return

    if cfg["make_all_knob_plots"]:
        make_all_knob_plots(cfg)
        return

    if cfg["replot"]:
        if cfg["mode"] == "fitted_R":
            if cfg["epoch"] == "all":
                epoch_list = list(range(1, len(EPOCH_GROUPS[cfg["source"]]) + 1))
            else:
                epoch_list = [int(cfg["epoch"])]
            for ep in epoch_list:
                replot_fitted_R(cfg["source"], ep, cfg, fixed)
        else:
            replot_dynamical(cfg["source"], cfg, fixed)
        return

    if cfg["mode"] == "fitted_R":
        if cfg["epoch"] == "all":
            epoch_list = list(range(1, len(EPOCH_GROUPS[cfg["source"]]) + 1))
        else:
            epoch_list = [int(cfg["epoch"])]
        for ep in epoch_list:
            run_fitted_R(cfg["source"], ep, cfg, fixed)

    elif cfg["mode"] == "dynamical":
        run_dynamical(cfg["source"], cfg, fixed)

    elif cfg["mode"] == "multi_epoch":
        run_multi_epoch(cfg["source"], cfg, fixed)

    else:
        raise ValueError(f"Unknown mode '{cfg['mode']}'")


if __name__ == "__main__":
    main()


# =====================================================================
# CHEAT SHEET -- every command below is safe to copy/paste. Swap
# --source wpp <-> dbl to match. --dir is a shared per-CONFIG label, not
# per-source: run2.slurm/run3.slurm/etc. all write BOTH sources under the
# same --dir (e.g. everything from run2.slurm lands under .../run2/,
# distinguished by the source name already baked into each file's own
# name, like wpp_fittedR_ep1_chain.h5 vs dbl_fittedR_ep1_chain.h5). So use
# --dir run2 (not run2_wpp) for wpp AND for dbl below.
# Nothing below this line executes; it's here so you don't have to
# remember the flags.
# =====================================================================
#
# --- 0. ALWAYS DO THIS FIRST for any new config before a real run ---
# NOTE: the printed estimate is SERIAL-equivalent (one likelihood eval,
# no multiprocessing) x nwalkers x nsamples -- your actual wall-clock time
# will be well under this since the pool evaluates all nwalkers each step
# in parallel. Useful for comparing configs against each other, not as a
# literal ETA.
# python runSampler.py --time_only --mode fitted_R --source wpp --epoch 1 \
#     --nwalkers 24 --nsamples 1e5
# python runSampler.py --time_only --mode dynamical --source wpp \
#     --nwalkers 24 --nsamples 1e5
#
# --- 1. RUN A FIT ---
# # single epoch, fitted_R (independent per-epoch fit: a, log10R, BG, log10n0)
# python runSampler.py --mode fitted_R --source wpp --epoch 3 \
#     --nwalkers 24 --nsamples 1e5 --ncores 12 --dir run2
#
# # every epoch of a source in one local call (fitted_R only; on SLURM,
# # submit one array task per epoch instead -- see run2.slurm/run3.slurm/...)
# python runSampler.py --mode fitted_R --source wpp --epoch all \
#     --nwalkers 24 --nsamples 1e5 --ncores 12 --dir run2
#
# --- 1b. DIFFERENT FIXED-PARAM / FREE-PARAM CONFIGS (run2, run3, ...) --
#     without editing FIXED_PARAMS/FREE_PARAMS_FITTED_R in the file ---
# # run2: eps_B fixed at 0.01 instead of 0.1
# python runSampler.py --mode fitted_R --source wpp --epoch all \
#     --nwalkers 24 --nsamples 1e5 --ncores 12 --dir run2 --eps_B 0.01
#
# # run3: eps_e fixed at 0.01 instead of 0.1 (eps_B back to default 0.1)
# python runSampler.py --mode fitted_R --source wpp --epoch all \
#     --nwalkers 24 --nsamples 1e5 --ncores 12 --dir run3 --eps_e 0.01
#
# # run4: eps_B FREE instead of fixed (5 free params now: a, log10R, BG,
# #        log10n0, log10eps_B). Physicality (eps_e+eps_B+eps_T<=1) is
# #        checked dynamically per-sample, no extra flag needed.
# python runSampler.py --mode fitted_R --source wpp --epoch all \
#     --nwalkers 24 --nsamples 1e5 --ncores 12 --dir run4 \
#     --free_params a,log10R,BG,log10n0,log10eps_B
#
# # run5: same as run4, plus eps_e fixed at 0.01
# python runSampler.py --mode fitted_R --source wpp --epoch all \
#     --nwalkers 24 --nsamples 1e5 --ncores 12 --dir run5 \
#     --free_params a,log10R,BG,log10n0,log10eps_B --eps_e 0.01
#
# (repeat all of the above with --source dbl, SAME --dir as the wpp run
# for that config -- e.g. --source dbl --dir run2, not run2_dbl)
#
# # dynamical (joint fit across ALL epochs of one source at once -- no
# # --epoch needed/used; much slower than fitted_R, time_only it first)
# python runSampler.py --mode dynamical --source wpp \
#     --nwalkers 24 --nsamples 1e5 --ncores 12 --dir run2
#
# # resume a run that got cut off (same nwalkers/dir as the original)
# python runSampler.py --mode fitted_R --source wpp --epoch 3 \
#     --nwalkers 24 --nsamples 1e5 --ncores 12 --dir run2 --resume
#
# --- 2. REPLOT (regenerate corner/trace/SED from an existing chain --
#     no resampling; use after tweaking plot formatting, not physics) ---
# python runSampler.py --replot --mode fitted_R --source wpp --epoch all \
#     --dir run2
# python runSampler.py --replot --mode dynamical --source wpp --dir run2
#
# --- 3. PARAMETER EVOLUTION + DENSITY PROFILE (needs all epochs of a
#     source already fit in fitted_R mode; both plots come from one call) ---
# python runSampler.py --make_evolution_plot --source wpp --dir run2
# python runSampler.py --make_evolution_plot --source dbl --dir run2
#
# --- 4. SED COLLAGE (grid of per-epoch data + Max Likelihood fit only;
#     needs all epochs of a source already fit in fitted_R mode) ---
# python runSampler.py --make_sed_collage --source wpp --dir run2
# python runSampler.py --make_sed_collage --source dbl --dir run2
#
# --- 5. KNOB/DIALS PLOT (SED sensitivity to each of the 11 params;
#     doesn't need any prior fit to exist) ---
# # centered on prior midpoints
# python runSampler.py --make_knob_plot --knob_source wpp --dir run2
#
# # centered on a real epoch's actual best fit (also uses that epoch's T)
# python runSampler.py --make_knob_plot --knob_source wpp --dir run2 \
#     --knob_from_summary mcmc_output/data/run2/wpp_fittedR_ep3/wpp_fittedR_ep3_summary.csv
#
# --- WHERE THINGS LAND ---
# mcmc_output/
# +-- data/<dir>/<tag>/          chain.h5, flatMCMC.csv, summary.csv (gitignore this)
# +-- plots/<dir>/<tag>/         trace.png, corner.png, sed.png, config.txt
# +-- plots/<dir>/               param_evolution.png, density_profile.png,
#                                 sed_collage.png, knob_plot_<source>.png
#                                 (these last ones live one level up from
#                                 per-epoch <tag> folders -- they're per-
#                                 source/run, not per-epoch)
