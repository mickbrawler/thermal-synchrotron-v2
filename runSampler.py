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


# =====================================================================
# SOURCE / EPOCH DEFINITIONS
# =====================================================================

# EDIT to match your machine / where you keep the raw data files.
_FILE_DIR = "/Users/michaelcamilo/research/tools_of_the_trade/files/"
_FILE_DIR = "/home/fs01/mc2923/thermal-synchrotron-v2"

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

def run_sampler(tag, rundir, ndim, labels, bounds, log_prob_fn, log_prob_args,
                 cfg):
    os.makedirs(rundir, exist_ok=True)
    backend_path = os.path.join(rundir, f"{tag}_chain.h5")
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

def save_chain_outputs(rundir, tag, sampler, labels, T=None):
    flat, flat_lp, discard, thin = get_flat_samples(sampler)

    # post-burn-in, thinned samples -- one row per posterior draw, columns
    # are the free params (in the untransformed/raw theta space used by
    # the sampler, e.g. log10R not R) plus log_prob.
    df = pd.DataFrame(flat, columns=labels)
    df["log_prob"] = flat_lp
    df.to_csv(os.path.join(rundir, f"{tag}_flatMCMC.csv"), index=False)

    # compact one-row-per-run summary (median, 68% interval, MAP best)
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
        os.path.join(rundir, f"{tag}_summary.csv"), index=False)

    print(f"    [{tag}] posterior summary:")
    for lab in labels:
        med = summary[f"{lab}_median"]
        print(f"      {lab}: {med:.4g}  (+{summary[f'{lab}_hi68']-med:.3g} "
              f"/ -{med-summary[f'{lab}_lo68']:.3g})  "
              f"best={summary[f'{lab}_best']:.4g}")

    raw_chain = sampler.get_chain()
    fig, axes = plt.subplots(len(labels), 1, figsize=(9, 2.2 * len(labels)),
                              sharex=True)
    for j, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(raw_chain[:, :, j], alpha=0.4, lw=0.6, color="steelblue")
        ax.set_ylabel(labels[j])
        ax.axvline(discard, color="crimson", ls="--", lw=1)
    axes[-1].set_xlabel("step")
    fig.suptitle(f"{tag} -- trace (red = discard boundary)")
    fig.tight_layout()
    fig.savefig(os.path.join(rundir, f"{tag}_trace.png"), dpi=130)
    plt.close(fig)

    fig = corner.corner(flat, labels=labels, show_titles=True,
                         title_fmt=".3g", quantiles=[0.16, 0.5, 0.84])
    fig.suptitle(tag, y=1.02)
    fig.savefig(os.path.join(rundir, f"{tag}_corner.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    return flat, flat_lp, summary


def plot_sed_fitted_R(rundir, tag, epoch_data, flat, flat_lp, fixed,
                       therm_el, pl_el, n_draws=60):
    nu_grid = np.logspace(8, 13, 400)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(epoch_data["freq"], epoch_data["flux"], yerr=epoch_data["eflux"],
                fmt="o", ms=5, color="k", zorder=5, label="Data")

    idx = np.random.choice(len(flat), size=min(n_draws, len(flat)), replace=False)
    for i in idx:
        Fnu = _fnu_fitted_R(flat[i], nu_grid, epoch_data["T"], epoch_data["z"],
                             epoch_data["d_L"], fixed, therm_el, pl_el)
        ax.plot(nu_grid, Fnu, color="steelblue", alpha=0.15, lw=1)

    best = flat[np.argmax(flat_lp)]
    Fnu_best = _fnu_fitted_R(best, nu_grid, epoch_data["T"], epoch_data["z"],
                              epoch_data["d_L"], fixed, therm_el, pl_el)
    ax.plot(nu_grid, Fnu_best, color="crimson", lw=2.5, label="MAP")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\nu$ (Hz)"); ax.set_ylabel(r"$F_\nu$ (Jy)")
    ax.set_title(tag)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(rundir, f"{tag}_sed.png"), dpi=130)
    plt.close(fig)


def plot_sed_dynamical(rundir, tag, epochs, flat, flat_lp, fixed,
                        therm_el, pl_el, n_draws=40):
    ncols = min(3, len(epochs))
    nrows = int(np.ceil(len(epochs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                              constrained_layout=True)
    ax_flat = np.atleast_1d(axes).reshape(-1)

    idx = np.random.choice(len(flat), size=min(n_draws, len(flat)), replace=False)
    best = flat[np.argmax(flat_lp)]
    nu_grid = np.logspace(8, 13, 300)

    for k, ep in enumerate(epochs):
        ax = ax_flat[k]
        ax.errorbar(ep["freq"], ep["flux"], yerr=ep["eflux"], fmt="o", ms=5,
                    color="k", zorder=5)
        for i in idx:
            Fnu = _fnu_dynamical(flat[i], nu_grid, ep["T"], ep["z"], ep["d_L"],
                                  fixed, therm_el, pl_el)
            ax.plot(nu_grid, Fnu, color="steelblue", alpha=0.15, lw=1)
        Fnu_best = _fnu_dynamical(best, nu_grid, ep["T"], ep["z"], ep["d_L"],
                                   fixed, therm_el, pl_el)
        ax.plot(nu_grid, Fnu_best, color="crimson", lw=2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"epoch {ep['epoch']} | T={ep['T']:.1f}d")
        ax.set_xlabel(r"$\nu$ (Hz)"); ax.set_ylabel(r"$F_\nu$ (Jy)")

    for k in range(len(epochs), len(ax_flat)):
        ax_flat[k].set_visible(False)

    fig.suptitle(tag)
    fig.savefig(os.path.join(rundir, f"{tag}_sed_allepochs.png"), dpi=130)
    plt.close(fig)


# =====================================================================
# EVOLUTION PLOT (aggregates per-epoch fitted_R summaries after the fact)
# =====================================================================

def make_evolution_plot(outdir, run_tag, source):
    rundir = os.path.join(outdir, run_tag)
    pattern = os.path.join(rundir, f"{source}_fittedR_ep*", f"{source}_fittedR_ep*_summary.csv")
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
        ax.plot(df["T"], best, "x--", color="steelblue", alpha=0.7, label="MAP")
        ax.set_ylabel(lab)
        if j == 0:
            ax.legend(fontsize=9)

    axes[-1].set_xlabel("T (observer-frame days)")
    fig.suptitle(f"{source} -- parameter evolution across epochs")
    fig.tight_layout()
    outpath = os.path.join(rundir, f"{source}_param_evolution.png")
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"    saved -> {outpath}")
    return df


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
    rundir = os.path.join(cfg["outdir"], cfg["run_tag"], tag)

    sampler = run_sampler(tag, rundir, len(PARAM_LABELS_FITTED_R),
                           PARAM_LABELS_FITTED_R, PRIORS_FITTED_R,
                           log_prob_fitted_R, args, cfg)

    flat, flat_lp, _ = save_chain_outputs(rundir, tag, sampler,
                                          PARAM_LABELS_FITTED_R, T=ep["T"])
    plot_sed_fitted_R(rundir, tag, ep, flat, flat_lp, fixed,
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
    rundir = os.path.join(cfg["outdir"], cfg["run_tag"], tag)

    sampler = run_sampler(tag, rundir, len(PARAM_LABELS_DYNAMICAL),
                           PARAM_LABELS_DYNAMICAL, PRIORS_DYNAMICAL,
                           log_prob_dynamical, args, cfg)

    flat, flat_lp, _ = save_chain_outputs(rundir, tag, sampler,
                                          PARAM_LABELS_DYNAMICAL)
    plot_sed_dynamical(rundir, tag, epochs, flat, flat_lp, fixed,
                        cfg["therm_el"], cfg["pl_el"])


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
