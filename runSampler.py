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
)

# --- which electron populations to include in the model ---
THERM_EL = True
PL_EL    = True

# --- priors for the free parameters (uniform on the given [lo, hi]) ---
# fitted_R mode fits: a, log10R, BG, log10n0
PRIORS_FITTED_R = dict(
    a       = (0.5, 4.0),
    log10R  = (14.0, 18.0),
    BG      = (0.01, 2.10),
    log10n0 = (-2.0, 6.0),
)
# dynamical mode fits: a, BG0, alpha, log10n0 (shared across all epochs)
# NOTE: alpha bounds are a rough placeholder -- tune these.
PRIORS_DYNAMICAL = dict(
    a       = (0.5, 4.0),
    BG0     = (0.01, 2.0),
    alpha   = (0.0, 3.0),
    log10n0 = (-2.0, 6.0),
)

# --- data handling ---
SNR_THRESHOLD       = 3.0
FLUX_ERR_FLOOR_FRAC = 0.10    # 10% error floor

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
    return f"{name} Joint Fit (All Epochs)"


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


def save_run_config_txt(plots_dir, tag, fixed, priors, therm_el, pl_el):
    lines = [f"Configuration for {tag}", "=" * (len(tag) + 15), "",
             "Fixed (non-fit) parameters:"]
    for k, v in fixed.items():
        lines.append(f"  {k:8s} = {v}")
    lines += ["", "therm_el = " + str(therm_el), "pl_el    = " + str(pl_el),
              "", "Priors (uniform) for free parameters:"]
    for k, (lo, hi) in priors.items():
        lines.append(f"  {k:8s} : ({lo}, {hi})")
    with open(os.path.join(plots_dir, f"{tag}_config.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


# =====================================================================
# DATA LOADING
# =====================================================================

def _load_wpp_raw():
    days, freq, flux, fluxErr, det = [], [], [], [], []

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
            except ValueError:
                continue

    order = np.argsort(days)
    return {
        "days":    np.array(days)[order],
        "freq":    np.array(freq)[order],
        "flux":    np.array(flux)[order],
        "fluxErr": np.array(fluxErr)[order],
        "det":     np.array(det, dtype=bool)[order],
    }


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
    mask = (t_rest >= t_lo) & (t_rest <= t_hi) & raw["det"]

    T = float(np.mean(raw["days"][mask]))
    flux_ep = raw["flux"][mask]
    eflux_ep = np.maximum(raw["fluxErr"][mask], FLUX_ERR_FLOOR_FRAC * flux_ep)
    freq_hz = raw["freq"][mask] * 1e9

    return dict(freq=freq_hz, flux=flux_ep, eflux=eflux_ep, T=T, z=z, d_L=d_L,
                epoch=epoch_idx, source=source)


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


def check_physicality(fixed):
    """eps_e + eps_B + eps_T <= 1 must hold (fraction of energy budget)."""
    total = fixed["eps_e"] + fixed["eps_B"] + fixed["eps_T"]
    if total > 1.0:
        raise ValueError(
            f"eps_e + eps_B + eps_T = {total:.3f} > 1.0 -- not physical. "
            f"Adjust FIXED_PARAMS."
        )


# =====================================================================
# LIKELIHOODS + LOG-PROB (module level, for picklability with mp spawn)
# =====================================================================

PARAM_LABELS_FITTED_R  = ["a", "log10R", "BG", "log10n0"]
PARAM_LABELS_DYNAMICAL = ["a", "BG0", "alpha", "log10n0"]


def _fnu_fitted_R(theta, freq, T, z, d_L, fixed, therm_el, pl_el):
    a, log10R, BG, log10n0 = theta
    R = 10.0 ** log10R
    n0 = 10.0 ** log10n0
    Lnu = flux_variables.LOS_IHG_Fitted_R(
        freq, fixed["s"], a, fixed["delta"], R, T, n0,
        fixed["eps_e"], fixed["eps_B"], fixed["eps_T"], fixed["p"],
        fixed["mu_u"], fixed["mu_e"], BG, fixed["k"], d_L, z,
        therm_el=therm_el, pl_el=pl_el,
    )
    return Lnu / (4.0 * np.pi * d_L ** 2) / C.Jy


def log_prob_fitted_R(theta, freq, flux, eflux, T, z, d_L, fixed,
                       therm_el, pl_el, bounds):
    lp = log_prior_box(theta, PARAM_LABELS_FITTED_R, bounds)
    a = theta[0]
    if not (0.5 < a < _a_upper_bound(fixed["p"], fixed["delta"])):
        return -np.inf
    if not np.isfinite(lp):
        return -np.inf

    try:
        Fnu_model = _fnu_fitted_R(theta, freq, T, z, d_L, fixed, therm_el, pl_el)
        chi2 = np.sum(((flux - Fnu_model) / eflux) ** 2)
        ll = -0.5 * chi2
    except Exception:
        return -np.inf

    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


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
                       therm_el, pl_el, n_draws=60):
    # Evaluate a bit wider than the data span so the curve visibly extends
    # "to the left and right" of the fit; natural_xy_limits then trims the
    # DISPLAYED window to where there's actually signal (+ padding), rather
    # than a fixed multi-decade span.
    freq_data = epoch_data["freq"]
    nu_grid = np.logspace(np.log10(freq_data.min()) - 1.0,
                           np.log10(freq_data.max()) + 1.0, 400)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(freq_data, epoch_data["flux"], yerr=epoch_data["eflux"],
                fmt="o", ms=5, color="k", zorder=5, label="Data")

    # Blue curves = SEDs evaluated at `n_draws` random DRAWS FROM THE FULL
    # POST-BURN-IN POSTERIOR (not a chi2-thresholded ensemble like the old
    # least-squares script's "keep everything within Delta-chi2" approach).
    # Because MCMC naturally spends more steps in high-likelihood regions,
    # denser/more-overlapping blue curves already reflect higher posterior
    # density -- no extra chi2 filtering needed on top.
    idx = np.random.choice(len(flat), size=min(n_draws, len(flat)), replace=False)
    curves = []
    for i in idx:
        Fnu = _fnu_fitted_R(flat[i], nu_grid, epoch_data["T"], epoch_data["z"],
                             epoch_data["d_L"], fixed, therm_el, pl_el)
        curves.append(Fnu)
        ax.plot(nu_grid, Fnu, color="steelblue", alpha=0.15, lw=1)

    best = flat[np.argmax(flat_lp)]
    Fnu_best = _fnu_fitted_R(best, nu_grid, epoch_data["T"], epoch_data["z"],
                              epoch_data["d_L"], fixed, therm_el, pl_el)
    curves.append(Fnu_best)
    # NOTE: "Max Likelihood" and "MAP" are the same point here specifically
    # because the priors are flat/uniform within bounds (log_prior is a
    # constant wherever finite) -- if you ever switch to non-uniform
    # priors, these would no longer coincide and this label should change.
    ax.plot(nu_grid, Fnu_best, color="crimson", lw=2.5, label="Max Likelihood")

    xlim, ylim = natural_xy_limits(nu_grid, curves, freq_data=freq_data,
                                    flux_data=epoch_data["flux"])
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
        nu_grid = np.logspace(np.log10(freq_data.min()) - 1.0,
                               np.log10(freq_data.max()) + 1.0, 300)
        ax.errorbar(freq_data, ep["flux"], yerr=ep["eflux"], fmt="o", ms=5,
                    color="k", zorder=5)
        curves = []
        # Blue curves here are also random draws from the joint posterior
        # (same note as plot_sed_fitted_R applies).
        for i in idx:
            Fnu = _fnu_dynamical(flat[i], nu_grid, ep["T"], ep["z"], ep["d_L"],
                                  fixed, therm_el, pl_el)
            curves.append(Fnu)
            ax.plot(nu_grid, Fnu, color="steelblue", alpha=0.15, lw=1)
        Fnu_best = _fnu_dynamical(best, nu_grid, ep["T"], ep["z"], ep["d_L"],
                                   fixed, therm_el, pl_el)
        curves.append(Fnu_best)
        ax.plot(nu_grid, Fnu_best, color="crimson", lw=2, label="Max Likelihood")

        xlim, ylim = natural_xy_limits(nu_grid, curves, freq_data=freq_data,
                                        flux_data=ep["flux"])
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
# EVOLUTION PLOT (aggregates per-epoch fitted_R summaries after the fact)
# =====================================================================

def make_evolution_plot(outdir, run_tag, source):
    data_rundir = os.path.join(outdir, "data", run_tag)
    plots_rundir = os.path.join(outdir, "plots", run_tag)
    os.makedirs(plots_rundir, exist_ok=True)

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

    fig, axes = plt.subplots(len(PARAM_LABELS_FITTED_R), 1,
                              figsize=(7, 2.5 * len(PARAM_LABELS_FITTED_R)),
                              sharex=True)
    for j, (ax, lab) in enumerate(zip(np.atleast_1d(axes), PARAM_LABELS_FITTED_R)):
        med = df[f"{lab}_median"].values
        lo  = df[f"{lab}_lo68"].values
        hi  = df[f"{lab}_hi68"].values
        best = df[f"{lab}_best"].values
        yerr = np.vstack([med - lo, hi - med])
        ax.errorbar(df["T"], med, yerr=yerr, fmt="o", capsize=4,
                    color="crimson", label="median +/- 68%")
        ax.plot(df["T"], best, "x--", color="steelblue", alpha=0.7,
                label="Max Likelihood")
        ax.set_ylabel(pretty_label(lab))
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
    return df


# =====================================================================
# "TURNING KNOBS" SED SENSITIVITY COLLAGE
# =====================================================================
# 3x4 grid: top-left is the unchanged baseline SED; the other 11 panels
# each vary ONE parameter across KNOB_N_VALUES points (holding all others
# at the baseline/center), for the 4 fitted_R free params plus the 7
# fixed microphysical/geometric params (mu_u, mu_e excluded).

KNOB_ORDER = PARAM_LABELS_FITTED_R + ["s", "delta", "eps_e", "eps_T", "p", "k", "eps_B"]


def _knob_bounds_and_logspace():
    bounds = dict(PRIORS_FITTED_R)
    bounds.update(KNOB_BOUNDS_FIXED)
    logspace = dict(KNOB_LOGSPACE_FREE)
    logspace.update(KNOB_LOGSPACE_FIXED)
    return bounds, logspace


def _knob_center(cfg):
    """Center/baseline values for all 11 dials, plus T/z/d_L to evaluate at."""
    center = dict(FIXED_PARAMS)  # s, delta, eps_e, eps_T, p, mu_u, mu_e, k, eps_B
    z = SOURCE_INFO[cfg["knob_source"]]["z"]
    d_L = Planck18.luminosity_distance(z).cgs.value

    if cfg["knob_from_summary"]:
        df = pd.read_csv(cfg["knob_from_summary"])
        row = df.iloc[0]
        for lab in PARAM_LABELS_FITTED_R:
            center[lab] = row[f"{lab}_best"]
        T = float(row["T"]) if "T" in row else cfg["knob_T"]
    else:
        for lab in PARAM_LABELS_FITTED_R:
            center[lab] = float(np.mean(PRIORS_FITTED_R[lab]))
        T = cfg["knob_T"]

    return center, T, z, d_L


def _eval_knob_sed(nu, params, T, z, d_L, therm_el, pl_el):
    R = 10.0 ** params["log10R"]
    n0 = 10.0 ** params["log10n0"]
    Lnu = flux_variables.LOS_IHG_Fitted_R(
        nu, params["s"], params["a"], params["delta"], R, T, n0,
        params["eps_e"], params["eps_B"], params["eps_T"], params["p"],
        params["mu_u"], params["mu_e"], params["BG"], params["k"], d_L, z,
        therm_el=therm_el, pl_el=pl_el,
    )
    return Lnu / (4.0 * np.pi * d_L ** 2) / C.Jy


def make_knob_plot(cfg):
    center, T, z, d_L = _knob_center(cfg)
    bounds, logspace = _knob_bounds_and_logspace()
    nu_grid = np.logspace(8, 13, 400)

    fig, axes = plt.subplots(3, 4, figsize=(20, 13))
    ax_flat = axes.reshape(-1)
    cmap = plt.get_cmap("viridis")

    # top-left: unchanged baseline
    base_curve = _eval_knob_sed(nu_grid, center, T, z, d_L,
                                cfg["therm_el"], cfg["pl_el"])
    xlim, ylim = natural_xy_limits(nu_grid, [base_curve])
    ax0 = ax_flat[0]
    ax0.plot(nu_grid, base_curve, color="k", lw=2.5)
    ax0.set_xlim(xlim); ax0.set_ylim(ylim)
    ax0.set_xscale("log"); ax0.set_yscale("log")
    ax0.set_title("Baseline (unchanged)")
    ax0.set_xlabel(r"$\nu$ (Hz)"); ax0.set_ylabel(r"$F_\nu$ (Jy)")

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
            Fnu = _eval_knob_sed(nu_grid, trial, T, z, d_L,
                                  cfg["therm_el"], cfg["pl_el"])
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
    fig.suptitle(f"{name} -- SED Sensitivity to Each Parameter "
                 f"(T={T:.1f}d, {center_note})", y=1.01)
    fig.tight_layout()

    plots_dir = os.path.join(cfg["outdir"], "plots", cfg["run_tag"])
    os.makedirs(plots_dir, exist_ok=True)
    outpath = os.path.join(plots_dir, f"knob_plot_{cfg['knob_source']}.png")
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved -> {outpath}")


# =====================================================================
# TIMING UTILITY
# =====================================================================

def time_single_eval(log_prob_fn, args, n_reps=3):
    bounds = args[-1]
    theta_mid = np.array([np.mean(b) for b in bounds.values()])
    t0 = time.time()
    for _ in range(n_reps):
        log_prob_fn(theta_mid, *args)
    return (time.time() - t0) / n_reps


# =====================================================================
# RUNNERS
# =====================================================================

def run_fitted_R(source, epoch_idx, cfg, fixed):
    print(f"\n=== [fitted_R] {source} epoch {epoch_idx} ===")
    ep = get_epoch_data(source, epoch_idx)
    args = (ep["freq"], ep["flux"], ep["eflux"], ep["T"], ep["z"], ep["d_L"],
             fixed, cfg["therm_el"], cfg["pl_el"], PRIORS_FITTED_R)

    if cfg["time_only"]:
        t_eval = time_single_eval(log_prob_fitted_R, args)
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

    save_run_config_txt(plots_dir, tag, fixed, PRIORS_FITTED_R,
                        cfg["therm_el"], cfg["pl_el"])

    sampler = run_sampler(tag, data_dir, len(PARAM_LABELS_FITTED_R),
                           PARAM_LABELS_FITTED_R, PRIORS_FITTED_R,
                           log_prob_fitted_R, args, cfg)

    flat, flat_lp, _ = save_chain_outputs(data_dir, plots_dir, tag, title_str,
                                          sampler, PARAM_LABELS_FITTED_R, T=ep["T"])
    plot_sed_fitted_R(plots_dir, tag, title_str, ep, flat, flat_lp, fixed,
                       cfg["therm_el"], cfg["pl_el"])


def run_dynamical(source, cfg, fixed):
    print(f"\n=== [dynamical] {source} (joint, all epochs) ===")
    epochs = get_all_epochs(source)
    args = (epochs, fixed, cfg["therm_el"], cfg["pl_el"], PRIORS_DYNAMICAL)

    if cfg["time_only"]:
        t_eval = time_single_eval(log_prob_dynamical, args)
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

    if not os.path.exists(backend_path):
        print(f"    SKIP {tag}: no chain found at {backend_path}")
        return
    os.makedirs(plots_dir, exist_ok=True)

    old_config = os.path.join(plots_dir, f"{tag}_config.txt")
    if os.path.exists(old_config):
        print(f"    NOTE: {old_config} exists from the original run -- "
              f"compare it to current FIXED_PARAMS before trusting the SED panel.")
    else:
        print(f"    NOTE: no saved config for {tag} (pre-dates that feature) -- "
              f"using CURRENT FIXED_PARAMS/priors to draw the SED; verify "
              f"these match what was actually used for this run.")

    backend = HDFBackend(backend_path)
    ep = get_epoch_data(source, epoch_idx)

    save_run_config_txt(plots_dir, tag, fixed, PRIORS_FITTED_R,
                        cfg["therm_el"], cfg["pl_el"])
    flat, flat_lp, _ = save_chain_outputs(data_dir, plots_dir, tag, title_str,
                                          backend, PARAM_LABELS_FITTED_R, T=ep["T"])
    plot_sed_fitted_R(plots_dir, tag, title_str, ep, flat, flat_lp, fixed,
                       cfg["therm_el"], cfg["pl_el"])
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
    if os.path.exists(old_config):
        print(f"    NOTE: {old_config} exists from the original run -- "
              f"compare it to current FIXED_PARAMS before trusting the SED panels.")
    else:
        print(f"    NOTE: no saved config for {tag} (pre-dates that feature) -- "
              f"using CURRENT FIXED_PARAMS/priors to draw the SEDs; verify "
              f"these match what was actually used for this run.")

    backend = HDFBackend(backend_path)
    epochs = get_all_epochs(source)

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
    p.add_argument("--mode", default=None, choices=["fitted_R", "dynamical"])
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
    p.add_argument("--replot", action="store_true", default=None,
                   help="regenerate plots from an existing saved chain "
                        "(no resampling); use with --source/--mode/--epoch/--dir "
                        "same as the original run")
    p.add_argument("--make_knob_plot", action="store_true", default=None)
    p.add_argument("--knob_source", default=None, choices=["wpp", "dbl"])
    p.add_argument("--knob_T", type=float, default=None)
    p.add_argument("--knob_from_summary", default=None,
                   help="path to a *_summary.csv to center the free-param "
                        "dials on a real epoch's MAP fit (also uses its T)")
    p.add_argument("--knob_n_values", type=int, default=None)
    p.add_argument("--therm_el", dest="therm_el", action="store_true", default=None)
    p.add_argument("--no-therm_el", dest="therm_el", action="store_false")
    p.add_argument("--pl_el", dest="pl_el", action="store_true", default=None)
    p.add_argument("--no-pl_el", dest="pl_el", action="store_false")
    return p.parse_args()


def _resolve(cli_val, panel_val):
    return panel_val if cli_val is None else cli_val


def build_config():
    cli = parse_args()
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
        replot      = _resolve(cli.replot, REPLOT),
        make_knob_plot = _resolve(cli.make_knob_plot, MAKE_KNOB_PLOT),
        knob_source = _resolve(cli.knob_source, KNOB_SOURCE),
        knob_T = _resolve(cli.knob_T, KNOB_T),
        knob_from_summary = _resolve(cli.knob_from_summary, KNOB_FROM_SUMMARY),
        knob_n_values = _resolve(cli.knob_n_values, KNOB_N_VALUES),
        therm_el    = _resolve(cli.therm_el, THERM_EL),
        pl_el       = _resolve(cli.pl_el, PL_EL),
        check_convergence       = CHECK_CONVERGENCE,
        convergence_check_every = CONVERGENCE_CHECK_EVERY,
        convergence_ntau        = CONVERGENCE_NTAU,
        convergence_rtol        = CONVERGENCE_RTOL,
    )
    return cfg


def main():
    cfg = build_config()
    fixed = dict(FIXED_PARAMS)
    check_physicality(fixed)

    if cfg["make_evolution_plot"]:
        make_evolution_plot(cfg["outdir"], cfg["run_tag"], cfg["source"])
        return

    if cfg["make_knob_plot"]:
        make_knob_plot(cfg)
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

    else:
        raise ValueError(f"Unknown mode '{cfg['mode']}'")


if __name__ == "__main__":
    main()
