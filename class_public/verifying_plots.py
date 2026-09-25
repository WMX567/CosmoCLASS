import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from math import pi

font = {'size': 16, 'family':'STIXGeneral'}
axislabelfontsize='large'
matplotlib.rc('font', **font)
matplotlib.rcParams['legend.fontsize']='medium'
plt.rcParams["figure.figsize"] = [8.0,6.0]
var_array = np.linspace(0,1.11e-22,5)
var_num = len(var_array)

# parameters = np.load('dataset/neff_test_param.npz')
# for i in range(3):
#     print('Parameters %d'%i)
#     for key in parameters.keys():
#         print(key,parameters[key][i])

clTT = np.load('dataset/neff_test_tt.npz')
clee = np.load('dataset/neff_test_ee.npz')
clpp = np.load('dataset/neff_test_pp.npz')

kvec = np.logspace(-4,np.log10(3),1000)
legarray = []
twopi = 2.*pi
legarray = [str(i) for i in range(var_num)]

fig_EE, ax_EE = plt.subplots()
fig_TT, ax_TT = plt.subplots()
fig_PP, ax_PP = plt.subplots()

for i,var in enumerate(var_array[:3]):
   
    if i == 0:
        var_color = 'k'
        var_alpha = 1.
        legarray.append(r'ref. $\Lambda CDM$')
    else:
        var_color = 'r'
        var_alpha = 1.*i/(var_num-1.)

    ax_TT.semilogx(clTT['modes'][i],clTT['features'][i]*clTT['modes'][i]*(clTT['modes'][i]+1)/twopi,color=var_color,alpha=var_alpha,linestyle='-', label=legarray[i])
    ax_EE.semilogx(clee['modes'][i],clee['features'][i]*clee['modes'][i]*(clee['modes'][i]+1)/twopi,color=var_color,alpha=var_alpha,linestyle='-', label=legarray[i])
    ax_PP.semilogx(clpp['modes'][i],clpp['features'][i]*clpp['modes'][i]*(clpp['modes'][i]+1)/twopi,color=var_color,alpha=var_alpha,linestyle='-', label=legarray[i])


ax_TT.set_xlim([2,9000])
ax_TT.set_xlabel(r'$\ell$')
ax_TT.set_ylabel(r'$[\ell(\ell+1)/2\pi]  C_\ell^\mathrm{TT}$')
ax_TT.legend()
fig_TT.tight_layout()
fig_TT.savefig('verifying_TT.pdf')

ax_EE.set_xlim([2,9000])
ax_EE.set_xlabel(r'$\ell$')
ax_EE.set_ylabel(r'$[\ell(\ell+1)/2\pi]  C_\ell^\mathrm{EE}$')
ax_EE.legend()
fig_EE.tight_layout()
fig_EE.savefig('verifying_EE.pdf')

ax_PP.set_xlim([2,9000])
ax_PP.set_xlabel(r'$\ell$')
ax_PP.set_ylabel(r'$[\ell(\ell+1)/2\pi]  C_\ell^\mathrm{PP}$')
ax_PP.legend()
fig_PP.tight_layout()
fig_PP.savefig('verifying_PP.pdf')

