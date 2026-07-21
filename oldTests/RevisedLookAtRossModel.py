# Ross Model - parameter study and SED fitting for AT2024wpp / AT2026dbl.

import os
import flux_variables
import Constants as C

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

from astropy.cosmology import Planck18
from scipy.optimize import least_squares


# CONTROL PANEL


RUN_SWEEP = False
RUN_FITTING = True

SOURCE = "wpp" # "wpp" (AT2024wpp) or "dbl" (AT2026dbl)

SOURCE_INFO = {
    "wpp": {"z": 0.0868},
    "dbl": {"z": 0.19},
}

# Epoch bounds in rest-frame days. Comment out epochs to skip.
EPOCH_GROUPS = {
    "wpp": [
        #(15,  20),
        (25,  35),
        (35,  50),
        (60,  75),
        (100, 110),
        (110, 120),
        (170, 180),
        #(250, 300),
    ],
    "dbl": [
        #(1,   4),
        #(5,   8.5),
        (9,   12),
        (15,  18),
        (22,  27),
        #(33,  35),
    ],
}

SWEEP_CONFIG = {
    "a":     (0.5,  4.0),
    "BG":    (0.01, 2.0),
    "eps_B": (1e-4, 0.5),
    "n0":    (1e1,  1e5),
}
SWEEP_NVALS = 5

FIT_EPOCHS      = None
FIT_N_SUCCESS   = 10
FIT_MAX_ATTEMPTS = 2000   # hard cap on total attempts per epoch (regardless of successes)
FIT_DELTA_CHI2  = 10.0
FIT_MAX_NFEV    = 5000
FIT_SHOW_ERR    = True
FIT_MIN_KEEP    = 5

PLOTS_DIR = "/Users/michaelcamilo/research/project_inDepthShockModeling/thermal_synchrotron_v2/plots"
FILES_DIR = "/Users/michaelcamilo/research/project_inDepthShockModeling/thermal_synchrotron_v2/files"


# PARAMETER CONFIGURATION


NU_LOW  = 1e8
NU_HIGH = 1e13
NU_RES  = 500

THERM_EL = True
PL_EL    = True

BASE_PARAMS = {
    "s":     1e5,
    "a":     3.0,
    "delta": 1.0,
    "R":     1e16,
    "BG":    0.01,
    "n0":    1e3,
    "eps_e": 0.1,
    "eps_B": 0.1,
    "eps_T": 0.4,
    "p":     3.0,
    "k":     0,
    "alpha": 0,
    "mu_u":  0.62,
    "mu_e":  1.18,
    "T":     50,
    "d_L":   1e28,
    "z":     0,
}

PARAM_LABELS = ["a", "log10R", "BG", "log10n0", "log10eps_B"]
LB = np.array([0.5, 14.0, 0.01, -2.0, -5.0])
UB = np.array([4.0, 18.0, 2.10,  6.0, -0.3])
BOUNDS = (LB, UB)


# DATA LOADING


_FILE_DIR      = "/Users/michaelcamilo/research/tools_of_the_trade/files/"
_SNR_THRESHOLD = 3.0

def wpp():
    T0_MJD = 60579.37093750
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
            days.append(float(cols[6]) - T0_MJD)
            freq.append(float(cols[0]))
            flux.append(float(cols[2]) * 1e-3)
            fluxErr.append(float(cols[3]) * 1e-3)
            det.append(float(cols[5]) > _SNR_THRESHOLD)

    vla_path = os.path.join(_FILE_DIR, "wppVLA_subbands_epoch1_2_SbandIgnore.dat")
    with open(vla_path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) < 8:
                continue
            days.append(float(cols[7]) - T0_MJD)
            freq.append(float(cols[0]))
            flux.append(float(cols[3]) * 1e-3)
            fluxErr.append(float(cols[4]) * 1e-3)
            det.append(float(cols[6]) > _SNR_THRESHOLD)

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

def dbl():
    data = pd.read_csv(
        os.path.join(_FILE_DIR, "dbl.txt"),
        sep=r"\s+",
    )
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

def get_epoch_data(epoch, source=None):
    if source is None:
        source = SOURCE

    if source not in _raw_cache:
        _raw_cache[source] = wpp() if source == "wpp" else dbl()
    raw = _raw_cache[source]

    z   = SOURCE_INFO[source]["z"]
    d_L = Planck18.luminosity_distance(z).cgs.value if z > 0 else BASE_PARAMS["d_L"]

    t_lo, t_hi = EPOCH_GROUPS[source][epoch - 1]
    t_rest = raw["days"] / (1 + z)
    mask   = (t_rest >= t_lo) & (t_rest <= t_hi)
    mask  &= raw["det"]

    T = float(np.mean(raw["days"][mask]))

    flux_ep  = raw["flux"][mask]
    eflux_ep = np.maximum(raw["fluxErr"][mask], 0.10 * flux_ep)  # 10% error floor

    source_params = {**BASE_PARAMS, "T": T, "z": z, "d_L": d_L}
    freq_hz = raw["freq"][mask] * 1e9
    return source_params, freq_hz, flux_ep, eflux_ep


# MODEL


def compute_Lnu(freq, params, therm_el=True, pl_el=True):
    return flux_variables.LOS_IHG_Fitted_R(
        freq,
        params["s"], params["a"], params["delta"], params["R"],
        params["T"], params["n0"], params["eps_e"], params["eps_B"], params["eps_T"], params["p"],
        params["mu_u"], params["mu_e"], params["BG"], params["k"],
        params["d_L"], params["z"],
        therm_el=therm_el, pl_el=pl_el,
    )

def compute_Fnu(freq, params, therm_el=True, pl_el=True):
    Lnu = compute_Lnu(freq, params, therm_el=therm_el, pl_el=pl_el)
    return Lnu / (4 * np.pi * params["d_L"]**2) / C.Jy

def theta_to_params(theta, base_params):
    a, log10R, BG, log10n0, log10epsB = theta
    return {
        **base_params,
        "a":     float(a),
        "R":     10.0**float(log10R),
        "BG":    float(BG),
        "n0":    10.0**float(log10n0),
        "eps_B": 10.0**float(log10epsB),
    }


# PARAMETER SWEEP


_LOG_PARAMS = {"s", "R", "n0", "eps_e", "eps_B", "eps_T"}

def _sweep_values(lo, hi, n, param_name):
    if param_name in _LOG_PARAMS:
        return np.logspace(np.log10(lo), np.log10(hi), n)
    return np.linspace(lo, hi, n)

def plot_param_collage(sweep_config, nvals=5, base_params=None, therm_el=True, pl_el=True):
    if base_params is None:
        base_params = BASE_PARAMS

    params  = list(sweep_config.keys())
    nu_grid = np.logspace(np.log10(NU_LOW), np.log10(NU_HIGH), NU_RES)
    colors  = cm.plasma(np.linspace(0.1, 0.9, nvals))

    fig, axes = plt.subplots(1, len(params), figsize=(5 * len(params), 4),
                             constrained_layout=True)
    if len(params) == 1:
        axes = [axes]

    for ax, param_name in zip(axes, params):
        lo, hi = sweep_config[param_name]
        vals   = _sweep_values(lo, hi, nvals, param_name)

        for color, v in zip(colors, vals):
            p = {**base_params, param_name: v}
            try:
                Lnu = compute_Lnu(nu_grid, p, therm_el=therm_el, pl_el=pl_el)
            except ZeroDivisionError:
                print(f"  {param_name}={v:.3g} → ZeroDivisionError, skipping")
                continue
            ax.loglog(nu_grid, Lnu, lw=2, alpha=0.85, color=color, label=f"{v:.3g}")

        ax.set_title(param_name, fontsize=11)
        ax.set_xlabel(r"$\nu$ (Hz)", fontsize=10)
        ax.set_ylabel(r"$L_\nu$ (erg s$^{-1}$ Hz$^{-1}$)", fontsize=10)
        ax.legend(title=param_name, fontsize=8, title_fontsize=8)

    return fig, axes


# FITTING


def _residuals(theta, base_params, freq, flux, eflux, therm_el, pl_el):
    params = theta_to_params(theta, base_params)
    Fnu    = compute_Fnu(freq, params, therm_el=therm_el, pl_el=pl_el)
    return (Fnu - flux) / eflux

def multistart_fit(epoch, n_success=200, bounds=BOUNDS,
                   therm_el=True, pl_el=True, max_nfev=5000, max_attempts=2000):
    lb, ub = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    source_params, freq, flux, eflux = get_epoch_data(epoch)
    n_par = len(lb)

    thetas    = np.full((n_success, n_par), np.nan)
    chi2_vals = np.full(n_success, np.nan)
    got       = 0

    rng = np.random.default_rng()
    for attempt in range(max_attempts):
        if got >= n_success:
            break
        x0 = lb + rng.random(n_par) * (ub - lb)
        try:
            res = least_squares(
                _residuals, x0=x0, bounds=(lb, ub), method="trf",
                args=(source_params, freq, flux, eflux, therm_el, pl_el),
                max_nfev=max_nfev,
            )
            if res.success and np.all(np.isfinite(res.x)) and np.isfinite(res.cost):
                thetas[got]    = res.x
                chi2_vals[got] = float(np.sum(res.fun**2))
                got += 1
                if got % max(1, n_success // 10) == 0:
                    print(f"  Epoch {epoch}: {got}/{n_success} fits (attempt {attempt+1})")
        except Exception:
            pass

    if got < n_success:
        print(f"  Warning: epoch {epoch} — {got}/{n_success} fits after {attempt+1} attempts")
        thetas, chi2_vals = thetas[:got], chi2_vals[:got]

    ndof = max(len(freq) - n_par, 1)
    return dict(
        epoch=epoch,
        source_params=source_params,
        freq=freq, flux=flux, eflux=eflux,
        thetas=thetas, chi2=chi2_vals,
        red_chi2=chi2_vals / ndof, ndof=ndof,
        therm_el=therm_el, pl_el=pl_el,
        bounds=(lb, ub),
    )

def save_solutions(out, idx):
    os.makedirs(FILES_DIR, exist_ok=True)
    ep   = out["epoch"]
    rows = out["thetas"][idx]
    chi2 = out["chi2"][idx]
    red  = out["red_chi2"][idx]

    df = pd.DataFrame(rows, columns=PARAM_LABELS)
    df.insert(0, "epoch", ep)
    df["chi2"]     = chi2
    df["red_chi2"] = red
    df["is_best"]  = (chi2 == chi2.min())

    path = os.path.join(FILES_DIR, f"{SOURCE}_ep{ep}_solutions.csv")
    df.to_csv(path, index=False)
    print(f"  Saved → {path}")

def select_solutions(out, delta_chi2=10.0):
    chi2 = out["chi2"]
    ok   = np.isfinite(chi2) & (chi2 <= np.nanmin(chi2[np.isfinite(chi2)]) + delta_chi2)
    return np.where(ok)[0]

def summarize_solutions(out, idx):
    if len(idx) == 0:
        print("  No solutions selected.")
        return
    chi2 = out["chi2"][idx]
    best = idx[np.argmin(chi2)]
    phys = theta_to_params(out["thetas"][best], out["source_params"])
    print(f"  Kept {len(idx)} solutions | chi2 min/med/max = "
          f"{np.min(chi2):.3g} / {np.median(chi2):.3g} / {np.max(chi2):.3g}")
    print(f"  Best theta:    {dict(zip(PARAM_LABELS, out['thetas'][best]))}")
    print(f"  Best physical: a={phys['a']:.3g}, R={phys['R']:.3g}, "
          f"BG={phys['BG']:.3g}, n0={phys['n0']:.3g}, eps_B={phys['eps_B']:.3g}")


# PLOTTING


def plot_epoch_grid(epochs, source=None):
    if source is None:
        source = SOURCE

    ncols = min(3, len(epochs))
    nrows = int(np.ceil(len(epochs) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             constrained_layout=True)
    ax_flat = np.array(axes).reshape(-1)

    for k, ep in enumerate(epochs):
        ax = ax_flat[k]
        sp, freq_hz, flux, eflux = get_epoch_data(ep, source=source)
        ax.errorbar(freq_hz, flux, yerr=eflux, fmt="o", ms=5, color="k", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$\nu$ (Hz)", fontsize=10)
        ax.set_ylabel(r"$F_\nu$ (Jy)", fontsize=10)
        ax.set_title(f"Epoch {ep}  |  T = {sp['T']:.1f} d (obs)", fontsize=11)

    for k in range(len(epochs), len(ax_flat)):
        ax_flat[k].set_visible(False)

    fig.suptitle(f"{source}  —  epoch SEDs", fontsize=13)
    return fig, axes

def plot_epoch_fit(out, idx):
    sp     = out["source_params"]
    freq, flux, eflux = out["freq"], out["flux"], out["eflux"]
    thetas = out["thetas"][idx]
    chi2   = out["chi2"][idx]
    n_par  = len(PARAM_LABELS)
    bins   = min(30, max(10, int(np.sqrt(len(idx)))))
    pad     = 0.35  # dex of padding on each side of the data
    x_lo    = 10**(np.log10(freq.min()) - pad)
    x_hi    = 10**(np.log10(freq.max()) + pad)
    nu_grid = np.logspace(np.log10(x_lo), np.log10(x_hi), NU_RES)

    fig = plt.figure(figsize=(max(14, 3 * n_par), 8))
    gs  = fig.add_gridspec(2, n_par, height_ratios=[1, 2.5], hspace=0.45, wspace=0.35)

    for j in range(n_par):
        ax = fig.add_subplot(gs[0, j])
        ax.hist(thetas[:, j], bins=bins, color="steelblue", edgecolor="white", lw=0.4)
        ax.set_xlabel(PARAM_LABELS[j], fontsize=9)
        ax.set_ylabel("count" if j == 0 else "", fontsize=9)
        ax.tick_params(labelsize=8)

    ax_sed = fig.add_subplot(gs[1, :])
    ax_sed.errorbar(freq, flux, yerr=eflux, fmt="o", ms=5, color="k", zorder=5, label="Data")

    for i in idx:
        params = theta_to_params(out["thetas"][i], sp)
        Fnu    = compute_Fnu(nu_grid, params, out["therm_el"], out["pl_el"])
        ax_sed.plot(nu_grid, Fnu, lw=1, alpha=0.2, color="steelblue")

    best_i   = idx[np.argmin(chi2)]
    params   = theta_to_params(out["thetas"][best_i], sp)
    Fnu_best = compute_Fnu(nu_grid, params, out["therm_el"], out["pl_el"])
    ax_sed.plot(nu_grid, Fnu_best, lw=2.5, color="crimson", label="Best fit")

    ax_sed.set_xscale("log"); ax_sed.set_yscale("log")
    ax_sed.set_xlim(x_lo, x_hi)
    ax_sed.set_xlabel(r"$\nu$ (Hz)")
    ax_sed.set_ylabel(r"$F_\nu$ (Jy)")
    ax_sed.legend()

    fig.suptitle(
        f"{SOURCE}  |  Epoch {out['epoch']}  |  T = {sp['T']:.1f} d  |  "
        f"{len(idx)} solutions  |  chi2_min = {np.nanmin(chi2):.2g}  (ndof = {out['ndof']})",
        fontsize=11
    )
    return fig

def _weighted_quantile(x, q, w):
    x, w = np.asarray(x), np.asarray(w)
    s    = np.argsort(x)
    cdf  = np.cumsum(w[s]); cdf /= cdf[-1]
    return np.interp(np.atleast_1d(q), cdf, x[s])

def plot_param_evolution(outs, keeps, epochs, show_err=True, min_keep=10, weighted=True):
    n_ep  = len(epochs)
    n_par = len(PARAM_LABELS)
    t     = np.array([outs[ep]["source_params"]["T"] for ep in epochs], float)

    best = np.full((n_ep, n_par), np.nan)
    q16  = np.full((n_ep, n_par), np.nan)
    q50  = np.full((n_ep, n_par), np.nan)
    q84  = np.full((n_ep, n_par), np.nan)
    has_err = np.zeros(n_ep, dtype=bool)

    for k, ep in enumerate(epochs):
        out   = outs[ep]; idx = keeps[ep]
        chi2k = out["chi2"][idx]
        best[k] = out["thetas"][idx[np.argmin(chi2k)]]

        if show_err and len(idx) >= min_keep:
            w      = np.exp(-0.5 * (chi2k - np.min(chi2k))) if weighted else np.ones(len(chi2k))
            thetas = out["thetas"][idx]
            for j in range(n_par):
                q16[k, j], q50[k, j], q84[k, j] = _weighted_quantile(thetas[:, j], [0.16, 0.50, 0.84], w)
            has_err[k] = True

    fig, axes = plt.subplots(n_par, 1, figsize=(7, 2.5 * n_par), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(t, best[:, j], "x--", color="steelblue", alpha=0.7, label="Best fit")
        if show_err and np.any(has_err):
            tt   = t[has_err]
            y    = q50[has_err, j]
            yerr = np.vstack([y - q16[has_err, j], q84[has_err, j] - y])
            ax.errorbar(tt, y, yerr=yerr, fmt="o", capsize=4,
                        color="crimson", label="Weighted median ± 68%")
        ax.set_ylabel(PARAM_LABELS[j])
        if j == 0:
            ax.legend(fontsize=9)

    axes[-1].set_xlabel("T (days)")
    fig.tight_layout()
    return fig, axes


# MAIN


def _savefig(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

if __name__ == "__main__":

    epochs = FIT_EPOCHS if FIT_EPOCHS is not None else list(range(1, len(EPOCH_GROUPS[SOURCE]) + 1))

    if RUN_SWEEP:
        print("\n=== Parameter sweep ===")
        fig, axes = plot_param_collage(SWEEP_CONFIG, nvals=SWEEP_NVALS,
                                       therm_el=THERM_EL, pl_el=PL_EL)
        _savefig(fig, "sweep_collage")

    if RUN_FITTING:
        outs, keeps = {}, {}

        for ep in epochs:
            print(f"\n=== Epoch {ep} ===")
            outs[ep]  = multistart_fit(ep, n_success=FIT_N_SUCCESS, bounds=BOUNDS,
                                       therm_el=THERM_EL, pl_el=PL_EL,
                                       max_nfev=FIT_MAX_NFEV,
                                       max_attempts=FIT_MAX_ATTEMPTS)
            idx       = select_solutions(outs[ep], delta_chi2=FIT_DELTA_CHI2)
            keeps[ep] = idx
            summarize_solutions(outs[ep], idx)
            save_solutions(outs[ep], idx)

            fig_fit = plot_epoch_fit(outs[ep], idx)
            _savefig(fig_fit, f"{SOURCE}_ep{ep}_fit")

        fig_evo, _ = plot_param_evolution(outs, keeps, epochs,
                                          show_err=FIT_SHOW_ERR, min_keep=FIT_MIN_KEEP)
        _savefig(fig_evo, f"{SOURCE}_param_evolution")

    else:
        print("\n=== Epoch SED preview ===")
        fig, _ = plot_epoch_grid(epochs)
        _savefig(fig, f"{SOURCE}_epoch_seds")
