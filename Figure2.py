import numpy as np
from scipy import optimize
import matplotlib.pylab as plt
plt.ion()
import thermalsyn_v2 as thermalsyn
import util
import Constants as C

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
            bG_sh[i,j],Mdot[i,j],R_tmp,n_tmp,B_tmp,U_tmp = thermalsyn.solve_shock(Lnu[i,j],nu_pk[i,j],t, initial_guess=initial_guess,direct_derivative_calculation=False, regime='thick',epsilon_T=epsilon_T,epsilon_B=epsilon_B,epsilon_e=epsilon_e,p=p,f=f,ell_dec=ell_dec,mu=mu,mu_e=mu_e)
            #bG_sh[i,j],Mdot[i,j],R_tmp,n_tmp,B_tmp,U_tmp = thermalsyn.solve_shock(Lnu[i,j],nu_pk[i,j],t, initial_guess=initial_guess,direct_derivative_calculation=np.nan, regime='thick',epsilon_T=epsilon_T,epsilon_B=epsilon_B,epsilon_e=epsilon_e,p=p,f=f,ell_dec=ell_dec,mu=mu,mu_e=mu_e)

plt.ion()

plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.xlim([4e-1,1e3])
plt.ylim([1e23,1e33])

# plot the critical luminosity
ln, = plt.loglog(nut_vec/(5e9*C.day),L_crit,'-',color='indianred',linewidth=5,zorder=0)
# add text along this curve
util.text_on_line(ln,plt.gca(), txt=r'$L_{{\rm crit}}$', fontsize=14, loc=0.21, va='top')

# --- calculate L_th ---
bG = 0.5*( bG_sh**2 - 2.0 + ( bG_sh**4 + 5.0*bG_sh**2 + 4.0 )**0.5 )**0.5
Gamma = (1.0+bG**2)**0.5
# find the (normalized) frequency at which the thermal and power-law emissitivities equal one another
x_j = np.zeros_like(Lnu)
for i in range(np.size(Lnu[:,0])):
    for j in range(np.size(Lnu[0,:])):
        if Gamma[i,j]==1:
            # fix bG << 1 case where numerical accuracy fails
            Theta = (2.0/3.0)*epsilon_T*(9.0*mu*C.mp/(32.0*mu_e*C.me))*bG_sh[i,j]**2
        else:
            Theta = thermalsyn.Theta_fun(Gamma[i,j],epsilon_T=epsilon_T,mu=mu,mu_e=mu_e)
        x_j[i,j] = thermalsyn.find_xj(Theta,delta=delta,p=p,z_cool=np.inf)
# calculate normalized peak frequency, x_pk
nu_Theta = thermalsyn.nu_Theta(bG_sh,Mdot,t,epsilon_T=epsilon_T,epsilon_B=epsilon_B,ell_dec=ell_dec,mu=mu,mu_e=mu_e)
x_pk = nu_pk/nu_Theta
# find contour along which nu_pk = nu_j (this is the definition of L_th)
cs = plt.contour(nut/(5e9*C.day),Lnu, np.log10(x_pk/x_j),[0.0], colors='dimgrey',linewidths=5,linestyles='-',zorder=0)
# remove this contour from the figure, and replot it as a line
L_th_x = cs.allsegs[0][0][:,0]
L_th_y = cs.allsegs[0][0][:,1]
for coll in cs.collections:
    coll.remove()
ln_th, = plt.loglog(L_th_x,L_th_y,'-',color='dimgrey',linewidth=5,zorder=0)
# add text along this line
util.text_on_line(ln_th,plt.gca(), txt=r'$L_{{\rm th}}$', fontsize=14, loc=0.15, va='top')
util.text_on_line(ln_th,plt.gca(), txt=r'thermal $e^{-}$'+'\n', fontsize=10, loc=0.35, va='top', arrow={'dist':3.5})

# plot contours of the shock proper-velocity
cs = plt.contour(nut/(5e9*C.day),Lnu, np.log10(bG_sh),np.log10(np.array([0.01,0.03,0.1,0.3])), colors='k',linewidths=1,linestyles='--',zorder=2)
plt.clabel(cs,inline=True,fontsize=9,fmt={l:str(util.round_to_1(10**l)) for l in cs.levels})
# create seperate contour with special label for bG_sh = 1
cs = plt.contour(nut/(5e9*C.day),Lnu, np.log10(bG_sh),np.log10(np.array([1])), colors='k',linewidths=1,linestyles='--',zorder=2)
plt.clabel(cs,inline=True,fontsize=9+1,fmt={l:r'$( \Gamma\beta )_{\rm sh} = $'+str(util.round_to_1(10**l)) for l in cs.levels})

# plot contours of the effective mass-loss rate
cs = plt.contour(nut/(5e9*C.day),Lnu, np.log10(Mdot/(C.Msun/C.yr/1e8)),[-8,-6,-4], colors='darkorchid',linewidths=1.5,linestyles='--',zorder=2)
plt.clabel(cs,inline=True,fontsize=9,fmt={l:r'$10^{'+str(util.round_to_1(l))+'}$' for l in cs.levels})
# create seperate contour with special label for Mdot = 1e-2 Msun / yr / 10^3 km/s
cs = plt.contour(nut/(5e9*C.day),Lnu, np.log10(Mdot/(C.Msun/C.yr/1e8)),[-2], colors='darkorchid',linewidths=1.5,linestyles='--',zorder=2)
plt.clabel(cs,inline=True,fontsize=9+1,manual=[(4e2,6e24)],fmt={l:r'$\dot{M}/v_{\rm w} = 10^{'+str(util.round_to_1(l))+r'} \, \frac{ M_\odot \, {\rm yr}^{-1} }{ 10^3\,{\rm km \, s}^{-1}}$' for l in cs.levels})

# load data for sample of radio transients
fl = 'Data/RadioTransients.txt'
nupk = np.genfromtxt(fl,usecols=4,skip_header=True)*1e9
Lpk = 4.0*np.pi*( np.genfromtxt(fl,usecols=5,skip_header=True)*1e-3*C.Jy )*( np.genfromtxt(fl,usecols=3,skip_header=True)*C.Mpc )**2
tpk = np.genfromtxt(fl,usecols=6,skip_header=True)*C.day
event = np.genfromtxt(fl,usecols=1,skip_header=True,dtype=str)
Type = np.genfromtxt(fl,usecols=2,skip_header=True,dtype=str)

# calculate L_th at the x-coordinates of the sample of radio transients
Lth_data = 10**np.interp( np.log10((nupk/5e9)*(tpk/C.day)), np.log10(L_th_x),np.log10(L_th_y) )

# plot radio transients
for i in range(np.size(Lpk)):
    marker = 'o'
    mrkrsz_small = 5
    mrkredgwdth_small = 1
    mrkrsz_large = 7
    mrkredgwdth_large = 2
    if Type[i]=='FBOT':
        clr='indianred'
        marker = '*'
        mrkrsz_small = 8
        mrkredgwdth_small = 1
        mrkrsz_large = 12
        mrkredgwdth_large = 1.5
    elif Type[i]=='GRBSN' or Type[i]=='Ic-BL':
        clr='goldenrod'
    elif Type[i]=='Ib' or Type[i]=='Ic' or Type[i]=='Ibc':
        clr='palevioletred'
    elif Type[i]=='II' or Type[i]=='IIb':
        clr='slateblue'
    elif Type[i]=='TDE':
        clr='darkblue'
        marker = 's'
        mrkrsz_large = 6.5
    # plot data point for transient number 'i'
    plt.plot((nupk[i]/5e9)*(tpk[i]/C.day),Lpk[i],marker,markersize=mrkrsz_large,color=clr,markerfacecolor='w',markeredgewidth=mrkredgwdth_large)
    # if peak luminosity falls above L_th or event is an FBOT then add text identifying event name
    if Lpk[i]>=Lth_data[i] or Type[i]=='FBOT':
        plt.text((nupk[i]/5e9)*(tpk[i]/C.day),Lpk[i],'\n'+event[i]+'  ',ha='right',va='center',fontsize=6,color=clr)

# create legend
Type_legend = [ 'FBOTs', 'SNe Ic-BL', 'SNe Ibc', 'SNe II', 'TDEs' ]
x = 0.55
y = 3e32
for i in range(np.size(Type_legend)):
    marker = 'o'
    mrkrsz_large = 7
    mrkredgwdth_large = 2
    if Type_legend[i]=='FBOTs':
        clr='indianred'
        marker = '*'
        mrkrsz_large = 12
        mrkredgwdth_large = 1.5
    elif Type_legend[i]=='SNe Ic-BL':
        clr='goldenrod'
    elif Type_legend[i]=='SNe Ibc':
        clr='palevioletred'
    elif Type_legend[i]=='SNe II':
        clr='slateblue'
    elif Type_legend[i]=='TDEs':
        clr='darkblue'
        marker = 's'
        mrkrsz_large = 6.5
    plt.plot(x, y/( 2.7**i ), marker,markersize=mrkrsz_large,color=clr,markerfacecolor='w',markeredgewidth=mrkredgwdth_large)
    plt.text(x, y/( 2.7**i ), '    '+Type_legend[i],ha='left',va='center',fontsize=11,color=clr)

# set axis limits, ticks, and labels
plt.gca().yaxis.set_ticks_position('both')
plt.xlim([4e-1,1e3])
plt.ylim([1e23,1e33])
plt.xlabel(r'$( \nu_{\rm pk} / 5\,{\rm GHz} ) \, ( t / {\rm day} )$',fontsize=12)
plt.ylabel(r'$L_{\nu_{\rm pk}} \,\,\,\,\, ({\rm erg \, s}^{-1}\,{\rm Hz}^{-1})$',fontsize=12)
plt.title(r'Synchrotron Self-Absorbed Peak ($\nu_{\rm pk} \simeq \nu_{\rm a}$)',fontsize=12)

plt.show()
