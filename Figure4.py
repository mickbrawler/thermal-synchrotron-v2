import numpy as np
from scipy import optimize
from scipy.optimize import curve_fit
import thermalsyn_v2 as thermalsyn
import Constants as C
import matplotlib.pylab as plt
plt.ion()
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# load data
Dlum = 38*C.Mpc
fl = 'Data/98bw_data.txt'
t = C.day*np.genfromtxt(fl,skip_header=True,usecols=0)
Fnu_data = 1e-3*C.Jy*np.genfromtxt(fl,skip_header=True,usecols=[1,2,3,4])
nu_data = np.array([1.38,2.49,4.80,8.64])*1e9

# limit to first 40 days
index = np.where(t>=40*C.day)[0]
t = np.delete(t,index)
Fnu_data = np.delete(Fnu_data,index,axis=0)

# calculate specific luminosity
Lnu_data = Fnu_data*4.0*np.pi*Dlum**2

# set microphysical and geometric parameters
epsilon_B = 0.1
epsilon_T = 0.4
epsilon_e = 1e-2
delta = epsilon_e/epsilon_T
f = 3.0/16.0
p = 2.3
ell_dec = 1.0

# open new figure
plt.figure()

# define colormap
cmap = plt.cm.viridis

# initiate frequency array
nu = np.logspace(np.log10(0.6),np.log10(20.0),100)*1e9

# initiate variables
bG_sh = np.zeros_like(t)
Mdot = np.zeros_like(t)

# iterate over time
for i in reversed(range(len(t))):

    # define a fitting function
    def fit_func(log10nu, bG_sh,log10Mdot):
        val = thermalsyn.Lnu_of_nu(bG_sh,(C.Msun/C.yr/1e8)*10**log10Mdot,10**log10nu,t[i], epsilon_T=epsilon_T,epsilon_B=epsilon_B,epsilon_e=epsilon_e,p=p,f=f,ell_dec=ell_dec)
        return np.log10(val)

    # fit to data
    popt, pcov = curve_fit(fit_func, np.log10(nu_data), np.log10(Lnu_data[i]), bounds=([0.05,-8.5],[1.2,-5.5]) )
    # save variables (note that Mdot here is in units of Msun / yr / 10^3 km/s)
    bG_sh[i] = popt[0]
    Mdot[i] = 10**popt[1]

    # get color from colormap
    clr = cmap(i/len(t))

    # plot data
    plt.loglog( nu_data/1e9, Fnu_data[i]/(1e-3*C.Jy), 'o',markersize=9,color=clr,markerfacecolor='w',markeredgewidth=3)

    # plot best-fit curve
    plt.loglog(nu/1e9, 10**fit_func(np.log10(nu), *popt)/(4.0*np.pi*Dlum**2)/(1e-3*C.Jy), '-',color=clr,linewidth=3)

# set axis limits, labels, and annotate
plt.xlim([0.6,20])
plt.ylim([2,100])
plt.xlabel(r'$\nu \,\,\, ({\rm GHz})$',fontsize=14)
plt.ylabel(r'$F_\nu \,\,\, ({\rm mJy})$',fontsize=14)
plt.gca().text(0.05,0.96,'SN1998bw',fontsize=16,horizontalalignment='left',verticalalignment='top',transform=plt.gca().transAxes)

# plot colorbar
norm = mpl.colors.Normalize(vmin=t[0]/C.day, vmax=t[-1]/C.day)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = plt.colorbar(sm)
cb.set_label(label=r'$t \,\,\, ({\rm day})$',size=14)

# initiate figure inset
axins = inset_axes(plt.gca(), width='50%',height='30%',loc=4,borderpad=0.5)

# plot inferred shock velocity as a function of time in inset
axins.plot(t/C.day,bG_sh,'-k',zorder=0)
axins.scatter(t/C.day,bG_sh, c=t/C.day,cmap=cmap)

# set inset axis labels, ticks, and limits
axins.minorticks_on()
axins.xaxis.set_ticks_position('top')
axins.xaxis.set_label_position('top')
axins.set_xlim([10+0.1,40-0.1])
axins.set_ylim([0.88,1.16])
axins.set_xlabel(r'$t \,\,\, ({\rm day})$',fontsize=12)
axins.set_ylabel(r'$(\Gamma\beta)_{\rm sh}$',fontsize=12)

plt.show()
