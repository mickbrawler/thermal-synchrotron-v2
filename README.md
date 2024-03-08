# `Thermal + Non-thermal' Synchrotron Emission Model
This code accompanies the paper "The Peak Frequency and Luminosity of Synchrotron Emitting Shocks: from Non-Relativistic to Ultra-Relativistic Explosions" ([Margalit &amp; Quataert 2024](https://ui.adsabs.harvard.edu/abs/2021arXiv211100012M/abstract); hereafter MQ24). It implements the 'thermal + non-thermal' model presented in MQ24, allowing calculation of the emergent synchrotron emission from an electron population comprised of both a thermal (Maxwellian) component and a non-thermal (power-law) distribution, including the effects of synchrotron self-absorption. This is an extension of the model described by [Margalit &amp; Quataert (2021)](https://ui.adsabs.harvard.edu/abs/2021arXiv211100012M/abstract) (hereafter MQ21) which was implemented in a previous version of this code. The new model includes relativistic corrections and allow treatment of non-relativistic shocks ultra-relativistic shocks, and anything in between.

# Requirements
- numpy
- scipy
- matplotlib (only required for example scripts)

# File descriptions
- Module `thermalsyn.py` implements formalism of MQ24 (see also MQ21) and can be used to calculate the resulting synchrotron emission
- Module `Constants.py` includes physical constants used by the code
- Module `util.py` includes a few utility function that are used by the example scripts (this module is NOT required by `thermalsyn.py`)
- Script `Figure1.py` includes the code used to generate Figure 1 of MQ24
- Script `Figure2.py` includes the code used to generate Figure 2 of MQ24
- Script `Figure3.py` includes the code used to generate Figure 3 of MQ24
- Script `Figure4.py` includes the code used to generate Figure 4 of MQ24
- Script `Figure5.py` includes the code used to generate Figure 5 of MQ24
- Directory `\Data` includes files that are used by some of the example scripts above (it is NOT required by `thermalsyn.py` more generally)

# Tutorial
The primary goal of this code is to compute the specific luminosity or flux density resulting from a shock. In the following we will illustrate a few example use cases.

Start by importing relevant modules and defining some model parameters
```python
import numpy as np
from scipy.optimize import curve_fit
import thermalsyn_v2 as thermalsyn
import Constants as C
import matplotlib.pylab as plt
plt.ion()

# set microphysical parameters
epsilon_B = 1e-1
epsilon_T = 0.4
epsilon_e = 1e-2
p = 2.5
f = 3.0/16.0
```

## Calculating an SED
The code snipet below shows an example calculation of an SED (spectral energy distribution). After setting the physical variables of the shock, namely its proper-velocity bG_sh (if the shock velocity is beta_sh\*c where c is the speed of light, and Gamma_sh = (1.0-beta_sh\**2)**(-0.5) is the corresponding shock Lorentz factor, then beta_sh*Gamma_sh is the shock proper velocity) and effective mass-loss rate (\dot{M}/v_w) which is related to the upstream density ahead of the shock (see equation 4 of MQ24). The function `thermalsyn.Lnu_of_nu()` is then called to calculate the resulting specific luminosity for such a shock.
```python
# set physical parameters
bG_sh = 0.3
Mdot = 1e-4*(C.Msun/C.yr/1e8) # in g/cm
t = 150*C.day
# define array of frequency
nu = np.logspace(9,11,300)

# calculate the specific luminosity at frequencies nu
Lnu = thermalsyn.Lnu_of_nu(bG_sh,Mdot,nu,t,epsilon_T=epsilon_T,epsilon_B=epsilon_B,epsilon_e=epsilon_e,p=p)

# plot the SED
plt.figure()
plt.loglog(nu, Lnu, '-',color='k',linewidth=5)
plt.xlabel(r'$\nu \,\,\,\,\, ({\rm Hz})$',fontsize=16)
plt.ylabel(r'$L_\nu \,\,\,\,\, ({\rm erg \, s}^{-1}\,{\rm Hz}^{-1})$',fontsize=16)
```

## Calculating Light-Curves
Next we can calculate some example light-curves (luminosity as a function of time, at some specified frequency). Here we also show an example where we use flag `density_insteadof_massloss=True` in calling `thermalsyn.Lnu_of_nu()` to indicate that we are specifying the upstream number density instead of an effective mass-loss rate.
```python
# set physical parameters
bG_sh = 0.3
Mdot = 1e-4*(C.Msun/C.yr/1e8) # in g/cm
# set external number density instead of mass-loss rate
n = 1e0 # in cm^{-3}
# oberved frequency
nu = 5e9
# define array of time
t = C.yr*np.logspace(-1,2,300)

# calculate the specific luminosity at times t (note we've set density_insteadof_massloss=True to indicate that the second input parameter is the number density, not a mass-loss rate)
Lnu = thermalsyn.Lnu_of_nu(bG_sh,n,nu,t,density_insteadof_massloss=True,epsilon_T=epsilon_T,epsilon_B=epsilon_B,epsilon_e=epsilon_e,p=p)

# plot the light-curve
plt.figure()
plt.loglog(t/C.yr, Lnu, '-',color='k',linewidth=5)
plt.xlabel(r'$t \,\,\,\,\, ({\rm yr})$',fontsize=16)
plt.ylabel(r'$L_\nu \,\,\,\,\, ({\rm erg \, s}^{-1}\,{\rm Hz}^{-1})$',fontsize=16)

# now calculate another light-curve for a shock whose velocity changes as a function of time
bG_sh = 0.3*((t+3.0*C.yr)/(3.0*C.yr))**(-3.0/5.0)
Lnu2 = thermalsyn.Lnu_of_nu(bG_sh,n,nu,t,density_insteadof_massloss=True,epsilon_T=epsilon_T,epsilon_B=epsilon_B,epsilon_e=epsilon_e,p=p)
# plot the result
plt.loglog(t/C.yr, Lnu2, '-',color='r',linewidth=5)
```

## Inferring Physical Properties from (mock) Data
In the code below we create a mock data set and then fit the model to this data.
```python
# redshift
z = 0.01
# luminosity distance
Dlum = 40*C.Mpc

# shock physical properties
bG_sh = 0.6
Mdot = 1e-5*(C.Msun/C.yr/1e8)
t = 100*C.day

# define array of frequency
nu = np.logspace(8.5,11,1000)
# calculate the model flux density at frequencies nu
Fnu = thermalsyn.Fnu_of_nu(bG_sh,Mdot,nu,t,Dlum=Dlum,z=z,epsilon_T=epsilon_T,epsilon_B=epsilon_B,epsilon_e=epsilon_e,p=p,f=f)

# create a mock dataset from the model
nu_data = np.array([0.4,1.4,3.0,6.0,10.0,80.0])*1e9
Fnu_data = thermalsyn.Fnu_of_nu(bG_sh,Mdot,nu_data,t,Dlum=Dlum,z=z,epsilon_T=epsilon_T,epsilon_B=epsilon_B,epsilon_e=epsilon_e,p=p,f=f)
# add random variation
Fnu_data *= np.random.lognormal(sigma=0.25,size=np.size(Fnu_data))

# eyeballl estimate of the SED peak
Fnu_pk = 200e-3*C.Jy
nu_pk = 5e9*(1.0+z) # in source rest frame
Lnu_pk = 4.0*np.pi*Dlum**2*Fnu_pk/(1.0+z) # in source rest frame

# get physical solution for target Lnu_pk, nu_pk, and t
bG_sh_estimate,Mdot_estimate,R,n,B,U = thermalsyn.solve_shock(Lnu_pk,nu_pk,t,regime='thick',epsilon_T=epsilon_T,epsilon_B=epsilon_B,epsilon_e=epsilon_e,p=p,f=f)

# perform a fit to the mock data SED
fit_func = lambda log10_nu,x,y: np.log10(thermalsyn.Fnu_of_nu(x,10**y*(C.Msun/C.yr/1e8),10**log10_nu,t,Dlum=Dlum,z=z,epsilon_T=epsilon_T,epsilon_B=epsilon_B,epsilon_e=epsilon_e,p=p,f=f))
popt, pcov = curve_fit( fit_func, np.log10(nu_data), np.log10(Fnu_data) )
# best fit shock velocity and mass-loss rates
bG_sh_fit = popt[0]
Mdot_fit = 10**popt[1]*(C.Msun/C.yr/1e8)
# generate SED from the the best-fit model
Fnu_fit = thermalsyn.Fnu_of_nu(bG_sh_fit,Mdot_fit,nu,t,Dlum=Dlum,z=z,epsilon_T=epsilon_T,epsilon_B=epsilon_B,epsilon_e=epsilon_e,p=p,f=f)

# for comparison purposes, calculate the SED that would have resulted from a comparable model with only power-law electrons
epsilon_T = 1e-10
epsilon_e = 0.4
Fnu_powerlaw = thermalsyn.Fnu_of_nu(bG_sh_fit,Mdot_fit,nu,t,Dlum=Dlum,z=z,epsilon_T=epsilon_T,epsilon_B=epsilon_B,epsilon_e=epsilon_e,p=p,f=f,pure_powerlaw_gamma_m=True)

# plot the data and results
plt.figure()
plt.loglog(nu, Fnu_powerlaw/(1e-3*C.Jy), '--',color='grey',linewidth=4)
plt.errorbar(nu_data, Fnu_data/(1e-3*C.Jy), yerr=(np.exp(0.25)-1.0)*Fnu_data/(1e-3*C.Jy), linestyle='none',marker='o',color='k',markersize=10)
plt.loglog(nu, Fnu_fit/(1e-3*C.Jy), '-',color='r',linewidth=6)
plt.xlabel(r'$\nu \,\,\,\,\, ({\rm Hz})$',fontsize=16)
plt.ylabel(r'$F_\nu \,\,\,\,\, ({\rm mJy})$',fontsize=16)
```
