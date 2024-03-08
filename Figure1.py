import numpy as np
from scipy import optimize
import thermalsyn_v2 as thermalsyn
import Constants as C
import matplotlib.pylab as plt
plt.ion()
from matplotlib import gridspec

# initiate physical parameters
mu = 0.62
mu_e = 1.18
epsilon_B = 1e-1
epsilon_T = 0.4
epsilon_e = 1e-2
delta = epsilon_e/epsilon_T
p = 3.0
f = 3.0/16.0

z_cool = np.inf

# open new figure
plt.figure(figsize=(16,4.8))
gs = gridspec.GridSpec(1,3)



# ------ first (left) subplot ------
plt.subplot(gs[0])

# add arrow and text showing direction of increasing shock velocity
plt.annotate('',xy=(0.65-0.04,0.962),xytext=(0.2-0.04,0.962),arrowprops=dict(facecolor='black',shrink=0.06),xycoords='figure fraction',fontsize=1)
plt.text(0.65-0.04-0.018,0.962,'shock velocity',transform=plt.gcf().transFigure,fontsize=16,ha='left',va='center')

# set desired peak properties
Lnu_pk = 1e28
nu_pk = 5e9
t = 150*C.day

bG_sh,Mdot,R,n,B,U = thermalsyn.solve_shock_analytic(Lnu_pk,nu_pk,t,regime='thick',epsilon_T=epsilon_T,epsilon_B=epsilon_B,epsilon_e=epsilon_e,p=p,f=f,mu=mu,mu_e=mu_e)
# get physical solution for target nu_pk, Lnu_pk, and t
bG_sh,Mdot,R,n,B,U = thermalsyn.solve_shock(Lnu_pk,nu_pk,t,regime='thick',epsilon_T=epsilon_T,epsilon_B=epsilon_B,epsilon_e=epsilon_e,p=p,f=f,mu=mu,mu_e=mu_e)
# print shock velocity
print('bG_sh = '+str(bG_sh))

# define array of frequency
nu = np.logspace(6,14,1000)

# calculate the specific luminosity at frequencies nu
Lnu = thermalsyn.Lnu_of_nu(bG_sh,Mdot, nu, t, epsilon_T=epsilon_T,epsilon_B=epsilon_B,epsilon_e=epsilon_e,p=p,f=f,mu=mu,mu_e=mu_e)

# calculate post-shock gas velocity
bG = 0.5*( bG_sh**2 - 2.0 + ( bG_sh**4 + 5.0*bG_sh**2 + 4.0 )**0.5 )**0.5
Gamma = ( 1.0 + bG**2 )**0.5
# calculate downstream electron number density
ne_prime = mu_e*4.0*Gamma*n
# calculate downstream electron temperature
Theta = thermalsyn.Theta_fun(Gamma,epsilon_T=epsilon_T,mu=mu,mu_e=mu_e)
# calculate thermal synchrotron frequency nu_Theta and normalized frequency x
nu_Theta = thermalsyn.nu_Theta(bG_sh,Mdot,t,epsilon_T=epsilon_T,epsilon_B=epsilon_B,mu=mu,mu_e=mu_e)
x = nu/nu_Theta

# calculate the optically-thin contribution due to thermal electrons
opticallythin_th = (16.0*np.pi**2/3.0)*R**3*f*thermalsyn.jnu_th(x,ne_prime,B,Theta,z_cool=z_cool)
# calculate the optically-thin contribution due to power-law electrons
opticallythin_pl = (16.0*np.pi**2/3.0)*R**3*f*thermalsyn.jnu_pl(x,ne_prime,B,Theta,delta=delta,p=p,z_cool=z_cool)

# plot the optically-thin components
plt.loglog(nu, opticallythin_th, '-',color='lightgrey',linewidth=3)
plt.loglog(nu, opticallythin_pl, '--',color='lightgrey',linewidth=3)

# find SED peak
Lpk_num = np.max(Lnu)
nupk_num = nu[Lnu==Lpk_num]
# plot vertical curve marking nu_pk and label with text
plt.loglog( nupk_num*np.array([1,1]), np.array([1e20,Lpk_num]), ':',color='k',linewidth='2')
plt.text( nupk_num, 0.02, '  '+r'$\nu_{\rm pk} \simeq \nu_{\rm a}$', transform=plt.gca().get_xaxis_transform(),ha='left',va='bottom',fontsize=16,color='k')

# plot vertical curve marking nu_Theta
plt.loglog( nu_Theta*np.array([1,1]), np.array([1e20,opticallythin_th[nu>=nu_Theta][0]]), ':',color='lightgrey',linewidth='2')

# plot the SED
plt.loglog(nu, Lnu, '-',color='k',linewidth=7)

# set axis limits, labels and ticks
plt.xlim([2e7,8e12])
plt.ylim([1e24,1e32])
plt.xlabel(r'$\nu \,\,\,\,\, ({\rm Hz})$',fontsize=16)
plt.ylabel(r'$L_\nu \,\,\,\,\, ({\rm erg \, s}^{-1}\,{\rm Hz}^{-1})$',fontsize=16)
plt.gca().yaxis.set_ticks_position('both')
plt.gca().tick_params(axis='y',direction='inout')

# label and annotate the subplot
plt.gca().text(1.0,1.022,'Synchrotron Self-Absorbed Peak',fontsize=18,horizontalalignment='center',transform=plt.gca().transAxes)
plt.gca().text(0.5,0.96,'(a)',fontsize=18,horizontalalignment='center',verticalalignment='top',transform=plt.gca().transAxes,weight='bold')
plt.gca().text(0.75,0.96,r'$L_{\nu_{\rm pk}} < L_{\rm th}$',fontsize=18,horizontalalignment='center',verticalalignment='top',transform=plt.gca().transAxes)
plt.gca().text(0.75,0.96,'\n\n'+'(peak dominated'+'\n'+r'  by power-law $e^{-}$)',fontsize=10,horizontalalignment='center',verticalalignment='top',transform=plt.gca().transAxes)

# annotate the spectral slopes
plt.gca().text(1.3e11,9e26,r'$\nu^{-\frac{p-1}{2}}$',fontsize=14,horizontalalignment='left',verticalalignment='bottom')
plt.gca().text(2.3e9,4e27,r'$\nu^{5/2}$',fontsize=14,horizontalalignment='right',verticalalignment='bottom')



# ------ second (middle) subplot ------
plt.subplot(gs[1])

# set desired peak properties
Lnu_pk = 2e29
nu_pk = 5e9
t = 70*C.day

# get physical solution for target nu_pk, Lnu_pk, and t
bG_sh,Mdot,R,n,B,U = thermalsyn.solve_shock(Lnu_pk,nu_pk,t,regime='thick',epsilon_T=epsilon_T,epsilon_B=epsilon_B,epsilon_e=epsilon_e,p=p,f=f,mu=mu,mu_e=mu_e)
# print shock velocity
print('bG_sh = '+str(bG_sh))

# define array of frequency
nu = np.logspace(6,14,1000)

# calculate the specific luminosity at frequencies nu
Lnu = thermalsyn.Lnu_of_nu(bG_sh,Mdot, nu, t, epsilon_T=epsilon_T,epsilon_B=epsilon_B,epsilon_e=epsilon_e,p=p,f=f,mu=mu,mu_e=mu_e)

# calculate post-shock gas velocity
bG = 0.5*( bG_sh**2 - 2.0 + ( bG_sh**4 + 5.0*bG_sh**2 + 4.0 )**0.5 )**0.5
Gamma = ( 1.0 + bG**2 )**0.5
# calculate downstream electron number density
ne_prime = mu_e*4.0*Gamma*n
# calculate downstream electron temperature
Theta = thermalsyn.Theta_fun(Gamma,epsilon_T=epsilon_T,mu=mu,mu_e=mu_e)
# calculate thermal synchrotron frequency nu_Theta and normalized frequency x
nu_Theta = thermalsyn.nu_Theta(bG_sh,Mdot,t,epsilon_T=epsilon_T,epsilon_B=epsilon_B,mu=mu,mu_e=mu_e)
x = nu/nu_Theta

# calculate the optically-thin contribution due to thermal electrons
opticallythin_th = (16.0*np.pi**2/3.0)*R**3*f*thermalsyn.jnu_th(x,ne_prime,B,Theta,z_cool=z_cool)
# calculate the optically-thin contribution due to power-law electrons
opticallythin_pl = (16.0*np.pi**2/3.0)*R**3*f*thermalsyn.jnu_pl(x,ne_prime,B,Theta,delta=delta,p=p,z_cool=z_cool)

# plot the optically-thin components
plt.loglog(nu, opticallythin_th, '-',color='lightgrey',linewidth=3)
plt.loglog(nu, opticallythin_pl, '--',color='lightgrey',linewidth=3)

# find SED peak
Lpk_num = np.max(Lnu)
nupk_num = nu[Lnu==Lpk_num]
# plot vertical curve marking nu_pk and label with text
plt.loglog( nupk_num*np.array([1,1]), np.array([1e20,Lpk_num]), ':',color='k',linewidth='2')
plt.text( nupk_num, 0.02, '  '+r'$\nu_{\rm pk} \simeq \nu_{\rm a}$', transform=plt.gca().get_xaxis_transform(),ha='left',va='bottom',fontsize=16,color='k')

# plot vertical curve marking nu_Theta and label with text
plt.loglog( nu_Theta*np.array([1,1]), np.array([1e20,opticallythin_th[nu>=nu_Theta][0]]), ':',color='lightgrey',linewidth='2')
plt.text( nu_Theta, 0.02, '  '+r'$\nu_\Theta$', transform=plt.gca().get_xaxis_transform(),ha='left',va='bottom',fontsize=16,color='silver')

# plot the SED
plt.loglog(nu, Lnu, '-',color='k',linewidth=7)

# set axis limits, labels and ticks
plt.xlim([2e7,8e12])
plt.ylim([1e24,1e32])
plt.xlabel(r'$\nu \,\,\,\,\, ({\rm Hz})$',fontsize=16)
plt.gca().yaxis.set_ticks_position('both')
plt.gca().tick_params(axis='y',direction='inout')
plt.gca().set_yticklabels([])

# label and annotate subplot
plt.gca().text(0.5,0.96,'(b)',fontsize=18,horizontalalignment='center',verticalalignment='top',transform=plt.gca().transAxes,weight='bold')
plt.gca().text(0.75,0.96,r'$L_{\nu_{\rm pk}} > L_{\rm th}$',fontsize=18,horizontalalignment='center',verticalalignment='top',transform=plt.gca().transAxes)
plt.gca().text(0.75,0.96,'\n\n'+'(peak dominated'+'\n'+r'  by thermal $e^{-}$)',fontsize=10,horizontalalignment='center',verticalalignment='top',transform=plt.gca().transAxes)

# annotate the spectral slopes
plt.gca().text(3.7e8,2.8e27,r'$\nu^2$',fontsize=14,horizontalalignment='right',verticalalignment='bottom')
plt.gca().text(1.8e10,5e28,r'$\nu^{5/6} e^{-\frac{3}{2} \left( 2 \nu/\nu_\Theta \right)^{1/3}}$',fontsize=14,horizontalalignment='left',verticalalignment='bottom')
plt.gca().text(3e11,4e26,r'$\nu^{-\frac{p-1}{2}}$',fontsize=14,horizontalalignment='left',verticalalignment='bottom')



# ------ third (right) subplot ------
plt.subplot(gs[2])

# set physical variables of shock
Mdot = 1e-6*(C.Msun/C.yr/1e8)
bG_sh = 3.0
tpk = 100*C.day
print('bG_sh = '+str(bG_sh))

# calculate other physical variables
bG = 0.5*( bG_sh**2 - 2.0 + ( bG_sh**4 + 5.0*bG_sh**2 + 4.0 )**0.5 )**0.5
Gamma = ( 1.0 + bG**2 )**0.5
R = (1.0+bG_sh**2)**0.5*bG_sh*C.c*t
n = Mdot/(4.0*np.pi*mu*C.mp*R**2)
B = ( 8.0*np.pi*epsilon_B*4.0*Gamma*(Gamma-1.0)*n*mu*C.mp*C.c**2 )**0.5

# define array of frequency
nu = np.logspace(6,14,1000)

# calculate the specific luminosity at frequencies nu
Lnu = thermalsyn.Lnu_of_nu(bG_sh,Mdot, nu, t, epsilon_T=epsilon_T,epsilon_B=epsilon_B,epsilon_e=epsilon_e,p=p,f=f,mu=mu,mu_e=mu_e)

# calculate post-shock gas velocity
bG = 0.5*( bG_sh**2 - 2.0 + ( bG_sh**4 + 5.0*bG_sh**2 + 4.0 )**0.5 )**0.5
Gamma = ( 1.0 + bG**2 )**0.5
# calculate downstream electron number density
ne_prime = mu_e*4.0*Gamma*n
# calculate downstream electron temperature
Theta = thermalsyn.Theta_fun(Gamma,epsilon_T=epsilon_T,mu=mu,mu_e=mu_e)
# calculate thermal synchrotron frequency nu_Theta and normalized frequency x
nu_Theta = thermalsyn.nu_Theta(bG_sh,Mdot,t,epsilon_T=epsilon_T,epsilon_B=epsilon_B,mu=mu,mu_e=mu_e)
x = nu/nu_Theta

# calculate the optically-thin contribution due to thermal electrons
opticallythin_th = (16.0*np.pi**2/3.0)*R**3*f*thermalsyn.jnu_th(x,ne_prime,B,Theta,z_cool=z_cool)
# calculate the optically-thin contribution due to power-law electrons
opticallythin_pl = (16.0*np.pi**2/3.0)*R**3*f*thermalsyn.jnu_pl(x,ne_prime,B,Theta,delta=delta,p=p,z_cool=z_cool)

# plot the optically-thin components
plt.loglog(nu, opticallythin_th, '-',color='lightgrey',linewidth=3)
plt.loglog(nu, opticallythin_pl, '--',color='lightgrey',linewidth=3)

# find SED peak
Lpk_num = np.max(Lnu)
nupk_num = nu[Lnu==Lpk_num]
# plot vertical curve marking nu_pk and label with text
plt.loglog( nupk_num*np.array([1,1]), np.array([1e20,Lpk_num]), ':',color='k',linewidth='2')
plt.text( nupk_num, 0.02, '  '+r'$\nu_{\rm pk} \simeq \nu_\Theta$', transform=plt.gca().get_xaxis_transform(),ha='left',va='bottom',fontsize=16,color='k')

# calculate absorption coefficient (in post-shock frame)
alpha_prime = thermalsyn.alphanu_th(x,ne_prime,B,Theta,z_cool=z_cool) + thermalsyn.alphanu_pl(x,ne_prime,B,Theta,delta=delta,p=p,z_cool=z_cool)
# calculate the optical depth as a function of frequency
tau = alpha_prime*( (4.0*f/3.0)*R/Gamma )

# find the synchrotron self-absorption frequency
nu_a = nu[np.log10(tau)<=0][0]
# plot vertical curve marking nu_a and label with text
plt.loglog( nu_a*np.array([1,1]), np.array([1e20,Lnu[nu>=nu_a][0]]), ':',color='k',linewidth='2')
plt.text( nu_a, 0.02, '  '+r'$\nu_{\rm a}$', transform=plt.gca().get_xaxis_transform(),ha='left',va='bottom',fontsize=16,color='k')

# plot the SED
plt.loglog(nu, Lnu, '-',color='k',linewidth=7)

# set axis limits, labels and ticks
plt.xlim([2e7,8e12])
plt.ylim([1e24,1e32])
plt.xlabel(r'$\nu \,\,\,\,\, ({\rm Hz})$',fontsize=16)
plt.gca().set_yticklabels([])
plt.gca().yaxis.set_ticks_position('both')
plt.gca().tick_params(axis='y',direction='inout')

# label and annotate subplot
plt.title('Optically-Thin Peak',fontsize=18)
plt.gca().text(0.5,0.96,'(c)',fontsize=18,horizontalalignment='center',verticalalignment='top',transform=plt.gca().transAxes,weight='bold')

# annotate the spectral slopes
plt.gca().text(5e7,1e29,r'$\nu^2$',fontsize=14,horizontalalignment='right',verticalalignment='bottom')
plt.gca().text(1.2e9,2.5e30,r'$\nu^{1/3}$',fontsize=14,horizontalalignment='right',verticalalignment='bottom')
plt.gca().text(1.3e11,1e30,r'$\nu^{5/6} e^{-\frac{3}{2} \left( 2 \nu/\nu_\Theta \right)^{1/3}}$',fontsize=14,horizontalalignment='left',verticalalignment='bottom')
plt.gca().text(1.8e12,5e27,r'$\nu^{-\frac{p-1}{2}}$',fontsize=14,horizontalalignment='left',verticalalignment='bottom')

# set figure layout
plt.gcf().tight_layout()
plt.subplots_adjust(wspace=0)

plt.show()
