import numpy as np
from scipy import optimize
import thermalsyn_v2 as thermalsyn
import Constants as C
import matplotlib.pylab as plt
plt.ion()


# initiate physical parameters
epsilon_B = 1e-1
epsilon_T = 0.4
epsilon_e = 1e-2
delta = epsilon_e/epsilon_T
p = 3.0
f = 3.0/16.0

# open new figure
plt.figure()

# define color palet
clr = 'goldenrod'
clr_face = 'w'
clr2 = 'grey'
clr_face2 = 'w'
clr3 = 'indianred'
clr_face3 = 'w'

# load data for sample of radio transients
fl = 'Data/RadioTransients.txt'
nupk = np.genfromtxt(fl,usecols=4,skip_header=True)*1e9
Lpk = 4.0*np.pi*( np.genfromtxt(fl,usecols=5,skip_header=True)*1e-3*C.Jy )*( np.genfromtxt(fl,usecols=3,skip_header=True)*C.Mpc )**2
tpk = np.genfromtxt(fl,usecols=6,skip_header=True)*C.day
event = np.genfromtxt(fl,usecols=1,skip_header=True,dtype=str)
Type = np.genfromtxt(fl,usecols=2,skip_header=True,dtype=str)

# remove jetted TDEs from sample in this figure
ind = np.where(Type=='TDE')[0]
nupk = np.delete(nupk,ind)
Lpk = np.delete(Lpk,ind)
tpk = np.delete(tpk,ind)
event = np.delete(event,ind)
Type = np.delete(Type,ind)

# calculate L_crit at the frequency and time coordinates of the sample of radio transients
Lcrit = np.zeros_like(Lpk)
for i in range(np.size(Lpk)):
    Lcrit[i] = thermalsyn.L_crit(nupk[i]*tpk[i],epsilon_T=epsilon_T,epsilon_B=epsilon_B,f=f)

# calculate L_th at the frequency and time coordinates of the sample of radio transients
Lth_data = 3e28*(epsilon_B/1e-1)**(-4.0/15.0)*(epsilon_T/0.4)**(-13.0/15.0)*(delta/0.025)**(8.0/15.0)*(nupk*tpk/(5e9*100*C.day))**(34.0/15.0)

# iterate over all events in sample
for i in range(np.size(Lpk)):
    # define marker shapes, sizes, and colors for different classes of events
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
        mrkrsz_large = 16
        mrkredgwdth_large = 2
    elif Type[i]=='GRBSN' or Type[i]=='Ic-BL':
        clr='goldenrod'
        mrkrsz_large = 9
        mrkredgwdth_large = 3
    elif Type[i]=='Ib' or Type[i]=='Ic' or Type[i]=='Ibc':
        clr='palevioletred'
    elif Type[i]=='II' or Type[i]=='IIb':
        clr='slateblue'

    # get physical properties of an event assuming standard power-law model
    bG_sh_pl,Mdot_pl,R_pl,n_pl,B_pl,U_pl = thermalsyn.solve_shock_analytic(Lpk[i],nupk[i],tpk[i],regime='thick',limit='pl',epsilon_T=epsilon_T,epsilon_B=epsilon_B,epsilon_e=epsilon_e,p=p,f=f)

    # solve for physical properties of a given event
    bG_sh,Mdot,R,n,B,U = thermalsyn.solve_shock(Lpk[i],nupk[i],tpk[i],regime='thick',direct_derivative_calculation=False,epsilon_T=epsilon_T,epsilon_B=epsilon_B,epsilon_e=epsilon_e,p=p,f=f)

    # plot properties that would have been inferred using the standard power-law model
    plt.plot([bG_sh,bG_sh_pl],[U,U_pl],':'+marker,markersize=mrkrsz_small,color='grey',markerfacecolor='w',markeredgewidth=mrkredgwdth_small,linewidth=1)

    # plot the inferred properties of events
    plt.plot(bG_sh,U,marker,markersize=mrkrsz_large,color=clr,markerfacecolor='w',markeredgewidth=mrkredgwdth_large)

    # for FBOTs, add text identifying name of event
    if Lpk[i]>=Lth_data[i] or Type[i]=='FBOT':
        plt.text(bG_sh,U,'\n'+'   '+event[i],ha='left',va='center',fontsize=7,color=clr)

# define some characteristic time and frequency for the sample
t_characteristic = 100*C.day
nu_characteristic = 5e9

# calculate the velocity of the shock at the transition L_th and for the characteristic t and nu above
L_th = 3e28*(epsilon_B/1e-1)**(-4.0/15.0)*(epsilon_T/0.4)**(-13.0/15.0)*(delta/0.025)**(8.0/15.0)*(nu_characteristic*t_characteristic/(5e9*100*C.day))**(34.0/15.0)
bG_sh_th, Mdot_tmp,R_tmp,n_tmp,B_tmp,U_tmp = thermalsyn.solve_shock(L_th,nu_characteristic,t_characteristic,regime='thick',epsilon_T=epsilon_T,epsilon_B=epsilon_B,epsilon_e=epsilon_e,p=p,f=f)
# plot the result
plt.axvline(bG_sh_th,linewidth=3,color='k',linestyle='--',zorder=0)

# calculate the velocity of the shock at L_crit and for the characteristic t and nu above
bG_sh_crit = thermalsyn.bG_sh_crit(nu_characteristic*t_characteristic,epsilon_T=epsilon_T,epsilon_B=epsilon_B,f=f)
# plot the result
plt.axvline(bG_sh_crit,linewidth=3,color='k',linestyle='--',zorder=0)

# set axes scale, ticks, and limits
plt.xscale('log')
plt.yscale('log')
plt.ylim([1e45,3e51])
plt.xlim([1e-2,1e1])
plt.gca().set_xticklabels([0.01,0.01,0.1,1,10])

# add text labeling the SEDs corresponding to each regime
plt.text(6e-2,1.2e45,'SED (a)',color='k',fontsize=12,verticalalignment='bottom',horizontalalignment='center')
plt.text(0.98,1.2e45,'SED (b)',color='k',fontsize=12,verticalalignment='bottom',horizontalalignment='center')
plt.text(4.8,1.2e45,'SED (c)',color='k',fontsize=12,verticalalignment='bottom',horizontalalignment='center')

# label the curves delineating L_th and L_crit
plt.text(bG_sh_th*1.08,2.3e51,r'$L_{\nu_{\rm pk}} = L_{\rm th}$',fontsize=14,color='k',verticalalignment='top',rotation=-90)
plt.text(bG_sh_crit*1.08,2.3e51,r'$\left(\Gamma\beta\right)_{\rm sh, crit}$',fontsize=14,color='k',verticalalignment='top')

# create legend
Type_legend = [ 'FBOTs', 'SNe Ic-BL', 'SNe Ibc', 'SNe II' ]
x = (4.0/3.0)*1e-2
y = 1.3e50*12
for i in range(np.size(Type_legend)):
    marker = 'o'
    mrkrsz_large = 9
    mrkredgwdth_large = 3
    if Type_legend[i]=='FBOTs':
        clr='indianred'
        marker = '*'
        mrkrsz_large = 16
        mrkredgwdth_large = 2
    elif Type_legend[i]=='SNe Ic-BL':
        clr='goldenrod'
    elif Type_legend[i]=='SNe Ibc':
        clr='palevioletred'
    elif Type_legend[i]=='SNe II':
        clr='slateblue'
    plt.plot(x, y/( 2.3**i ), marker,markersize=mrkrsz_large,color=clr,markerfacecolor='w',markeredgewidth=mrkredgwdth_large)
    plt.text(x, y/( 2.3**i ), '    '+Type_legend[i],ha='left',va='center',fontsize=11,color=clr)

# set axis labels
plt.xlabel(r'$\left( \Gamma\beta \right)_{\rm sh}$',fontsize=14)
plt.ylabel(r'$U \,\,\,\,\, ({\rm erg})$',fontsize=14)

plt.show()
