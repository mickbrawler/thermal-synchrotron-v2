import numpy as np
from scipy import optimize
import thermalsyn_v2 as thermalsyn
import Constants as C
import matplotlib.pylab as plt
plt.ion()
import util

# initiate physical parameters
mu = 0.62
mu_e = 1.18
epsilon_B = 1e-1
epsilon_T = 0.4
epsilon_e = 1e-2
delta = epsilon_e/epsilon_T
p = 3.0
f = 3.0/16.0
ell_dec = 1.0

t = C.day
nu_pk_vec = 5e9*np.logspace(np.log10(0.4),3,38)
# create array of nu_pk*t (x-axis)
nut_vec = nu_pk_vec*t
# create array of Lnu_pk (y-axis)
Lnu_vec = np.logspace(23,33,44)

# create 2D arrays from x and y vectors
Lnu,nut = np.meshgrid(Lnu_vec,nut_vec)
nu_pk = nut/t

# initiate variables
bG_sh = np.nan*np.zeros_like(Lnu)
Mdot = np.nan*np.zeros_like(Lnu)

# calculate critical luminosity
L_crit = thermalsyn.L_crit(nut_vec,epsilon_T=epsilon_T,epsilon_B=epsilon_B,f=f,mu=mu,mu_e=mu_e)

# run over the 2D phase space
for i in range(np.size(Lnu[:,0])):
    # print crude progress bar
    print('{:.1f}%'.format(100.0*i/np.size(Lnu[:,0])))
    for j in range(np.size(Lnu[0,:])):
        # ignore region above the critical luminosity
        if Lnu[i,j] < L_crit[i]:
            # set initial guess for solution (use prior nearby solutions if available)
            if j>0:
                initial_guess = [ bG_sh[i,j-1], Mdot[i,j-1] ]
            elif i>0:
                initial_guess = [ bG_sh[i-1,j], Mdot[i-1,j] ]
            else:
                initial_guess = [np.nan,np.nan]
            # run solver
            bG_sh[i,j],Mdot[i,j],R_tmp,n_tmp,B_tmp,U_tmp = thermalsyn.solve_shock(Lnu[i,j],nu_pk[i,j],t, initial_guess=initial_guess,direct_derivative_calculation=True, regime='thin',epsilon_T=epsilon_T,epsilon_B=epsilon_B,epsilon_e=epsilon_e,p=p,f=f,ell_dec=ell_dec,mu=mu,mu_e=mu_e)

# open figure and set axis scale and limits
plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.xlim([4e-1,1e3])
plt.ylim([1e23,1e31])

# plot contours of the shock proper-velocity
cs = plt.contour(nut/(5e9*C.day),Lnu, np.log10(bG_sh),np.log10(np.array([10,30,100,300])), colors='k',linewidths=1,linestyles='--',zorder=2)
# label contours
fmt = {}
strs = ['10','30','100','300']
for l, s in zip(cs.levels, strs):
    fmt[l] = s
plt.clabel(cs,inline=True,fontsize=9,fmt=fmt)

# create seperate contour with special label for bG_sh = 3
cs = plt.contour(nut/(5e9*C.day),Lnu, np.log10(bG_sh),np.log10(np.array([3])), colors='k',linewidths=1,linestyles='--',zorder=2)
plt.clabel(cs,inline=True,fontsize=9+1,fmt={l:r'$( \Gamma\beta )_{\rm sh} = '+str(int(10**l))+r'$' for l in cs.levels})

# plot contours of the effective mass-loss rate
cs = plt.contour(nut/(5e9*C.day),Lnu, np.log10(Mdot/(C.Msun/C.yr/1e8)),[-12,-10,-8,-6], colors='darkorchid',linewidths=1.5,linestyles='--',zorder=2)
plt.clabel(cs,inline=True,fontsize=9,fmt={l:r'$10^{'+str(int(l))+r'}$' for l in cs.levels})

# plot the critical luminosity
ln, = plt.loglog(nut_vec/(5e9*C.day),L_crit,'-',color='indianred',linewidth=5)
# add text along this curve
util.text_on_line(ln,plt.gca(), txt=r'$L_{{\rm crit}}$', fontsize=14, loc=0.21, va='top')

# set axis limits, ticks, and labels
plt.xlim([4e-1,1e3])
plt.ylim([1e23,1e31])
plt.xlabel(r'$( \nu_{\rm pk} / 5\,{\rm GHz} ) \, ( t / {\rm day} )$',fontsize=12)
plt.ylabel(r'$L_{\nu_{\rm pk}} \,\,\,\,\, ({\rm erg \, s}^{-1}\,{\rm Hz}^{-1})$',fontsize=12)
plt.title(r'Optically-Thin Peak ($\nu_{\rm pk} \simeq \nu_\Theta$)',fontsize=12)

plt.show()
