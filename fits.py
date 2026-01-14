import numpy as np
from astropy.cosmology import Planck18
from astropy.table import Table
from scipy.optimize import curve_fit
#from thermal_synchrotron_v2 import thermalsyn_v2 as thermalsyn
#from thermal_synchrotron_v2 import Constants as C
import thermalsyn_v2 as thermalsyn
import flux_variables
import Constants as C
import matplotlib.pyplot as plt
import scipy.stats
import emcee
import corner
import pandas as pd
import math
import multiprocessing
import argparse
import os
import sys
plt.style.use('dark_background')
from sklearn.mixture import GaussianMixture
from numpy.linalg import norm

class Fits:

    def __init__(self, Dir, epoch=1):

        # Ross' defaults
        
        # We fit for these!
        self.a = 3.0 # Index for magnetic field probability distribution
        self.R = 1e16 # Undocumented... forward shock wave radius
        self.BG = 0.01 # Shock proper velocity at R=R0 (the radius at which the maximum perpendicular
                       # extent of the shock is reached; see Figure 1 in FM25)
        self.n0 = 1e3 # Nominal value for upstream number density
        self.eps_B = 0.1 # Fraction of local fluid energy in magnetic field

        # These stay constant
        self.s = 1e5 # Effective range of magnetic field values B1/B0
        self.delta = 1.0 # Index for explicit relation between post-shock number density and magnetic field strength

        self.eps_e = 0.000001 # Fraction of local fluid energy in power-law electrons
        self.eps_T = 0.4 # Fraction of local fluid energy in thermal electrons
        self.p = 3.0 # Power-law electron distribution index
        self.k = 0 # Power-law index for stratified density (Eq. 12 in FM25)
        self.alpha = 0 # Power-law index for deceleration (Eq. 11 in FM25)

        self.mu_u = 0.62 # Mean molecular weight; nominal value 0.62
        self.mu_e = 1.18 # Mean molecular weight per electron; nominal value 1.18

        self.nu_low, self.nu_high = 1e9,1e12
        self.nu_res = 500 # Frequency array length
        self.therm_el = True # If True---calculates thermal electron synchrotron flux
        self.pl_el = False # If True---calculates power-law electron synchrotron flux

        self.b = {
            "a":     (0, 6.0),
            "R":     (15, 17), # log(cm)
            "BG":    (0.01, 0.9),
            "n0":    (-2, 8), # log(cm^-3)
            "eps_B": (-6, 0) # log()
        }
 
        self.dir = Dir; self.epoch = epoch

    def AT2024wpp(self):

        self.name = "wpp"
        filename = "/Users/michaelcamilo/research/project_Thermal/data/wippet_radio.txt"
        data = pd.read_csv(filename, sep="\t", skiprows=1, names=[
               "FBOTName", "dt_days", "nu_GHz", "flux_mJy", "ferr_mJy", "rms_mJy",
               "det", "reference", "obs", "timestart", "timestop", "tmid"])
        Days = data['dt_days'].values # days
        Freq = data['nu_GHz'].values * 1e9 # Hz
        Flux = data['flux_mJy'].values * 1e-3 # Jy 
        eFlux = data['ferr_mJy'].values * 1e-3 # Jy

        epochGroups = [
            #[12.02],
            [28.91,32.9,36.653,33.684,34.714],
            [43.88,43.611,43.628,43.670],
            [70.78,71.596,71.614,71.702],
            [110.70, 113.445, 128.65],
            [191.5],
            #[298.25]
        ]

        epochIndex = self.epoch - 1
        group = np.array(epochGroups[epochIndex])
        tMask = np.any(np.isclose(Days[:, None], group[None, :], rtol=0, atol=1e-3), axis=1)
        self.freq = Freq[tMask]
        self.flux = Flux[tMask]
        self.fluxErr = eFlux[tMask]
        
        # Known info
        self.T = np.mean(Days[tMask]) # Time (days) of observation in observer's frame
        self.z = 0.0868 # Source redshift
        self.d_L = Planck18.luminosity_distance(self.z).cgs.value # Luminosity distance (cm)

    def compute_Lnu_R(self, params, therm_el, pl_el):
        nu = np.logspace(np.log10(params["nu_low"]), np.log10(params["nu_high"]), params["nu_res"])
        Lnu = flux_variables.LOS_IHG_Fitted_R(
            nu,
            params["s"], params["a"], params["delta"], params["R"],
            params["T"], params["n0"], params["eps_e"], params["eps_B"], params["eps_T"], params["p"],
            params["mu_u"], params["mu_e"],
            params["BG"], params["k"],
            params["d_L"], params["z"],
            therm_el=therm_el, pl_el=pl_el
        )
        return nu, Lnu
    
    def log_prob(self, theta):
        # Has logic from log_prior, log_likelihood, and log_probability

        aa, RR, BGBG, n0n0, eps_Beps_B = theta
        
        if self.b['a'][0]<aa<self.b['a'][-1] and self.b['R'][0]<RR<self.b['R'][-1] and \
           self.b['BG'][0]<BGBG<self.b['BG'][-1] and self.b['n0'][0]<n0n0<self.b['n0'][-1] and \
           self.b['eps_B'][0]<eps_Beps_B<self.b['eps_B'][-1]: lp = 0.0
        else: lp = -np.inf

        RR, n0n0, eps_Beps_B = 10**RR, 10**n0n0, 10**eps_Beps_B
        if self.eps_e + eps_Beps_B + self.eps_T > 1.0: return -np.inf

        params = dict(
            a=aa, R=RR, BG=BGBG, n0=n0n0, eps_B=eps_Beps_B,
            s=self.s, delta=self.delta, T=self.T, 
            eps_e=self.eps_e, eps_T=self.eps_T, p=self.p,
            k=self.k, alpha=self.alpha, mu_u=self.mu_u, 
            mu_e=self.mu_e, d_L=self.d_L, z=self.z,
            nu_res=self.nu_res, nu_low=self.nu_low, nu_high=self.nu_high,
        )

        nu_model, Lnu_model = self.compute_Lnu_R(params, self.therm_el, self.pl_el)
        Fnu_model = Lnu_model / (4*np.pi*self.d_L**2)
        Fnu_model = 10**np.interp(np.log10(self.freq),np.log10(nu_model),np.log10(Fnu_model)) # log log interpolation
        F_obs, F_exp = self.flux, Fnu_model/C.Jy

        # 10% error floor
        min_fluxErr = 0.1 * np.abs(F_obs)
        adj_fluxErr = np.maximum(self.fluxErr, min_fluxErr)

        chi2 = np.sum(((F_obs - F_exp) ** 2) / (adj_fluxErr ** 2))
        ll = -0.5 * chi2
        if not np.isfinite(ll): return -np.inf # for BG>1, model can produce nans
        if not np.isfinite(lp): return -np.inf # checks if np.inf is outputted
        return lp + ll

    def MCMC_fit(self, source, nwalkers=10, nsamples=10000, load=False, 
                 Med=True, ccorner=True, SED=True, overlay=True, 
                 evol=True, profile=True):

        ndim = 5
        self.labels = [r"a", r"R", r"$\beta_{sh}$", r"$n_0$", r"$\epsilon_B$"]

        if not load:
            source()

            p0 = []
            while len(p0) < nwalkers: # Initial walker positions

                trial = np.array([np.random.uniform(self.b['a'][0], self.b['a'][-1]),
                                  np.random.uniform(self.b['R'][0], self.b['R'][-1]),
                                  np.random.uniform(self.b['BG'][0], self.b['BG'][-1]),
                                  np.random.uniform(self.b['n0'][0], self.b['n0'][-1]),
                                  np.random.uniform(self.b['eps_B'][0], self.b['eps_B'][-1])])
                
                # Check if the sample passes the log_prob physicality check
                if np.isfinite(self.log_prob(trial)): p0.append(trial)

            p0 = np.array(p0)

            print(f"{self.name}: Epoch: {self.epoch}")
            # multiprocessing for cluster
            #with multiprocessing.get_context("fork").Pool(nwalkers) as pool:
            #    sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_prob, pool=pool)
            #    sampler.run_mcmc(p0, nsamples)

            # multiprocessing for mac
            with multiprocessing.get_context("spawn").Pool(processes=os.cpu_count()) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_prob, pool=pool)
                sampler.run_mcmc(p0, nsamples, progress=True)

            # Save raw chain
            samples = sampler.get_chain()
            os.makedirs(f"../data/{self.name}/{self.dir}", exist_ok=True) # make sure data dir exists
            np.save(f"../data/{self.name}/{self.dir}/{self.name}_epoch{self.epoch}_rawMCMC.npy", samples)

            # Post-processing: discard burn-in and thin
            tau = sampler.get_autocorr_time()
            print('Auto-corr time:', tau)
            Discard = int(np.max(tau) * 3)
            Thin = int(np.max(tau) / 2)
            flat_samples = sampler.get_chain(discard=Discard, thin=Thin, flat=True)

            # Evaluate likelihoods for accepted samples
            likelihoods = np.array([self.log_prob(sample) for sample in flat_samples])

            # Stack parameters and likelihoods into one array
            self.flat_samples = np.column_stack((flat_samples, likelihoods))
            np.savetxt(f"../data/{self.name}/{self.dir}/{self.name}_epoch{self.epoch}_MCMC.txt", self.flat_samples)