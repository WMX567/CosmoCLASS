import numpy as np
import math
from scipy.interpolate import interp1d, UnivariateSpline
#from scipy.misc import derivative
from scipy.signal import find_peaks
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# matplotlib.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.major.size'] = 12
plt.rcParams['xtick.minor.size'] = 8
plt.rcParams['ytick.major.size'] = 12
plt.rcParams['ytick.minor.size'] = 6

from numpy.fft import fft, ifft , rfft, irfft , fftfreq
from numpy import exp, log, log10, cos, sin, pi, cosh, sinh , sqrt
from classy import Class
from scipy.optimize import fsolve
from scipy.special import gamma
from scipy.special import hyp2f1
import sys,os
from time import time
from scipy.integrate import quad
import scipy.integrate as integrate
from scipy import special
from scipy.special import factorial

z_pk=3

IDM_bestfit = {# fixed LambdaCDM parameters
                   'ln10^{10}A_s':3.017714,
                   'n_s':9.446883e-01,
                   'tau_reio':7.427423e-02,
		   'log10_G_eff_nu':-6,
                   'omega_b':2.271890e-02,
                   'h':6.825509e-01,
#                   '100*theta_s':1.0431,
#                   'YHe':2.4798e-01,
#                   'N_eff':3.046,
                   'N_ur':2.0328,
                   'N_ncdm':1,
#                   'N_ncdm':0,
                   'm_ncdm':2.107045e-01,
                   # other output and precision parameters
#                    'P_k_max_1/Mpc':100.0,
                   'z_pk':z_pk,
                   'output':'mPk,tCl,pCl,lCl',
                   'lensing':'yes',
		   'perturbations_verbose':5,
		   'z_max_pk':4}
#                   'l_max_scalars':10000.}

M_bestfit = Class()
M_bestfit.set(IDM_bestfit)
#compute linear dmeff
M_bestfit.set({'non linear':'no',
        'omega_cdm':1.191894e-01,
      })
M_bestfit.compute()

IDM_bestfit = {'ln10^{10}A_s':3.017714,
               'n_s':9.446883e-01,
               'tau_reio':7.427423e-02,
		'log10_G_eff_nu':-1.722373,
		'N_ur':2.0328,
		'h':6.825509e-01,
		'N_ncdm':1,
		'm_ncdm':2.107045e-01,
		'omega_b':2.271890e-02,
		'omega_cdm':1.191894e-01,
		'perturbations_verbose':5,
		'output':'mPk',
                'z_max_pk':4}

M_bestfit_EFT = Class()
M_bestfit_EFT.set(IDM_bestfit)
#compute EFT
M_bestfit_EFT.compute()

third_bestfit = {'ln10^{10}A_s':3.017714,
                'n_s':9.446883e-01,
                'tau_reio':7.427423e-02,
                'log10_G_eff_nu':-3.5,
                'N_ur':2.0328,
                'h':6.825509e-01,
                'N_ncdm':1,
                'm_ncdm':2.107045e-01,
                'omega_b':2.271890e-02,
                'omega_cdm':1.191894e-01,
                'perturbations_verbose':5,
                'output':'mPk',
                'z_max_pk':4}

third = Class()
third.set(third_bestfit)
third.compute()

LCDM_bestfit = {# fixed LambdaCDM parameters
                   'ln10^{10}A_s':3.0448,
                   'n_s':0.96605,
                   'tau_reio':0.0543,
                   'omega_b':0.022383,
                   'h':0.6732,
#                   '100*theta_s':1.0431,
#                   'YHe':2.4798e-01,
#                   'N_eff':3.046,
                   'N_ur':2.0328,
                   'N_ncdm':1,
#                   'N_ncdm':0,
                   'm_ncdm':0.06,
                   # other output and precision parameters
#                    'P_k_max_1/Mpc':100.0,
                   'z_pk':z_pk,
                   'output':'mPk,tCl,pCl,lCl',
                   'lensing':'yes',
                   'z_max_pk':4}
#                   'l_max_scalars':10000.}

M_LCDM_bestfit = Class()
M_LCDM_bestfit.set(LCDM_bestfit)
#compute linear LCDM                                                                                                                                     
M_LCDM_bestfit.set({ 'non linear':'no',
        'omega_cdm':0.12011,
      })
M_LCDM_bestfit.compute()

M_LCDM_EFT_bestfit = Class()
M_LCDM_EFT_bestfit.set(LCDM_bestfit)
#compute EFT LCDM                                                                                                                                     
M_LCDM_EFT_bestfit.set({ 'non linear':'PT',
        'IR resummation':'Yes',
        'Bias tracers':'Yes',
        'cb':'No',
        'RSD':'Yes',
        'AP':'Yes',
        'Omfid':'0.31',
        'omega_cdm':0.12011
      })
M_LCDM_EFT_bestfit.compute()

LCDM_alldata = {'ln10^{10}A_s':3.049684,
		'n_s':9.584220e-01,
		'tau_reio':5.569188e-02,
		'omega_b':2.237307e-02,
		'h':6.671403e-01,
		'N_ur':2.0328,
		'N_ncdm':1,
		'm_ncdm':1.471747e-01,		
		'z_pk':z_pk,
		'output':'mPk,tCl,pCl,lCl',	
		'lensing':'yes',
		'z_max_pk':4,
		'non linear':'no',
		'omega_cdm':1.197181e-01}				

M_LCDM_alldata = Class()
M_LCDM_alldata.set(LCDM_alldata)
M_LCDM_alldata.compute()

#fig, axs = plt.subplots()
plt.xlabel('$k \ [h/\mathrm{Mpc}]$',fontsize=21, labelpad=12)
plt.xscale('log')
plt.ylabel(r'$P/P_{\Lambda \mathrm{CDM,} \ Planck}$', fontsize=21, labelpad=12)
plt.xlim([1e-3,100])
plt.minorticks_on()
#ax.set_ylim([-30,2])

#fig.setp(axs[0].get_xticklabels(), visible=True)

h_LCDM_bestfit = M_LCDM_bestfit.h()
k = np.linspace(log(0.001),log(100),10000)
k = np.exp(k)
khvec_LCDM_bestfit = k*h_LCDM_bestfit

power_spectrum_LCDM_bestfit = np.linspace(log(0.001),log(50),10000)
for i in range(len(k)):
        power_spectrum_LCDM_bestfit[i] = M_LCDM_bestfit.pk_lin(k[i]*h_LCDM_bestfit,z_pk)*h_LCDM_bestfit**3

M_LCDM_EFT_bestfit.initialize_output(khvec_LCDM_bestfit, z_pk, len(khvec_LCDM_bestfit))
power_spectrum_LCDM_EFT_bestfit = M_LCDM_EFT_bestfit.pk_mm_real(0.)

h = M_bestfit.h()
k = np.linspace(log(0.001),log(100),10000)
k = np.exp(k)
khvec = k*h

power_spectrum_bestfit = np.linspace(log(0.001),log(100),10000)
for i in range(len(k)):
        power_spectrum_bestfit[i] = M_bestfit.pk_lin(k[i]*h,z_pk)*h**3

h = M_bestfit_EFT.h()
khvec = k*h

power_spectrum_bestfit_EFT = np.linspace(log(0.001),log(100),10000)
for i in range(len(k)):
	power_spectrum_bestfit_EFT[i] = M_bestfit_EFT.pk_lin(k[i]*h, z_pk)*h**3
#power_spectrum_bestfit_EFT = M_bestfit_EFT.pk_mm_real(0.)

power_spectrum_third = np.linspace(log(0.001),log(100),10000)
for i in range(len(k)):
        power_spectrum_third[i] = third.pk_lin(k[i]*third.h(),z_pk)*third.h()**3

h = M_LCDM_alldata.h()
khvec = k*h

power_spectrum_LCDM_alldata = np.linspace(log(0.001),log(100),10000)
for i in range(len(k)):
	power_spectrum_LCDM_alldata[i] = M_LCDM_alldata.pk_lin(k[i]*h, z_pk)*h**3

k_new = []
bestfit_EFT_new = []
LCDM_EFT_bestfit_new = []

for i in range(len(k)):
    if k[i] <= 0.4:
        k_new.append(k[i])
        #bestfit_EFT_new.append(power_spectrum_bestfit_EFT[i])
        LCDM_EFT_bestfit_new.append(power_spectrum_LCDM_EFT_bestfit[i])

test = np.zeros(len(k))

arr = np.abs(k - 1)
index = np.argmin(arr)

amp = 0.2749576188092795
y = 0.31/amp*power_spectrum_bestfit_EFT[index]/power_spectrum_LCDM_bestfit[index]

#slope_iv = -12.716064086424145 

spl = UnivariateSpline(k, power_spectrum_bestfit_EFT, s=0)
spl_c = UnivariateSpline(k, power_spectrum_LCDM_bestfit, s=0)
spl_inter = UnivariateSpline(k, power_spectrum_third, s=0)
spl_weak = UnivariateSpline(k, power_spectrum_bestfit, s=0)
spl_ca = UnivariateSpline(k, power_spectrum_LCDM_alldata, s=0)

dP_dk = spl.derivative()
dP_dk_c = spl_c.derivative()
dP_dk_inter = spl_inter.derivative()
dP_dk_weak = spl_weak.derivative()
dP_dk_ca = spl_ca.derivative()

x1 = 0.999
x2 = 1.001
y1 = spl(x1)
y2 = spl(x2)
m = (y2-y1)/(x2-x1)
print(m)

xs = [x1, x2]
ys = [y1, y2]
ys_c = [spl_c(x1), spl_c(x2)]
m_c = (spl_c(x2)-spl_c(x1))/(x2-x1)

f_iv = interp1d(np.array(xs), np.array(ys))
f_cdm = interp1d(np.array(xs), np.array(ys_c))

slope_xs = [0.8, 1.3]

def axline(x, y, m):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array([0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1, 1.025, 1.05, 1.075, 1.1, 1.125, 1.15, 1.175, 1.2, 1.225, 1.25])
    y_vals = y + m * (x_vals - x) 
    plt.plot(10**(x_vals-1), y_vals, color = 'black', linestyle = '-', linewidth = 0.1)

#axs.plot(np.array(slope_xs), f_iv(np.array(slope_xs))/f_cdm(np.array(slope_xs)), color = 'black')

#axs.plot(k, power_spectrum_bestfit/power_spectrum_LCDM_bestfit, color = '#377eb8', label=r'$\Lambda$CDM bestfit')
line1, = plt.plot(k, power_spectrum_bestfit_EFT/power_spectrum_LCDM_bestfit, color = '#4daf4a', label=r'log_10(G_eff \ MeV^2) = -1.72',linewidth = 2.5,zorder=4)
line2, = plt.plot(k, power_spectrum_third/power_spectrum_LCDM_bestfit, color = '#ff7f00', label=r'log_10(G_eff \ MeV^2) = -3.5',linewidth = 2.5,zorder=3)
line3, = plt.plot(k, power_spectrum_bestfit/power_spectrum_LCDM_bestfit, color = '#377eb8', label=r'log_10(G_eff \ MeV^2) = -6',linewidth = 2.5,zorder=2)
line4, = plt.plot(k, power_spectrum_LCDM_alldata/power_spectrum_LCDM_bestfit, color = 'black', label=r'All data LambdaCDM best-fit', linewidth = 2.5,zorder=1)
#axs.axvline(1, color='black', linestyle='--')

#plt.errorbar(k[index], y, yerr=[[y - 0.29/amp*power_spectrum_bestfit_EFT[index]/power_spectrum_LCDM_bestfit[index]], [0.33/amp*power_spectrum_bestfit_EFT[index]/power_spectrum_LCDM_bestfit[index] - y]],color = 'black',marker = '_',capsize = 2)

line5 = plt.axvspan(0.0125,0.4,color='gray', alpha=.15, label=r'BOSS')
line6 = plt.axvspan(0.1,0.35,color='gray', alpha=.3, label=r'DES')
line7 = plt.axvline(1, color='black', linestyle='--', label=r'Lyman-alpha')

y = np.arange(10,20,0.1)

#plt.axvspan(0.0125,0.4,color='gray', alpha=.15)

#axs[1].plot(k, test, color = '#4daf4a')
#ax.axvspan(0.01, 0.4, alpha=0.1, color='gray')
#ax.text(0.035, -15, 'probed by BOSS', fontsize = 14)

first_legend = plt.legend(handles=[line1,line2,line3,line4], fontsize='12',ncol=1,loc='lower left')

ax = plt.gca().add_artist(first_legend)

plt.legend(handles=[line5,line6,line7], fontsize='12',ncol=1,loc='upper right')

plt.tight_layout()
plt.savefig('newest fig 1.pdf')

fig1, axs1 = plt.subplots()
axs1.plot(k, dP_dk(k)*(k/power_spectrum_bestfit_EFT), color = '#4daf4a', label=r'log_10(G_eff \ MeV^2) = -1.72',zorder=5,linewidth=2.5)
axs1.plot(k, dP_dk_inter(k)*(k/power_spectrum_third), color = '#ff7f00', label=r'log_10(G_eff \ MeV^2) = -3.5',zorder=4,linewidth=2.5)
axs1.plot(k, dP_dk_weak(k)*(k/power_spectrum_bestfit), color = '#377eb8', label=r'log_10(G_eff \ MeV^2) = -6',zorder=2,linewidth=2.5)
#axs1.plot(k, dP_dk_c(k)*(k/power_spectrum_LCDM_bestfit), color = '#999999', label=r'Planck $\Lambda$CDM best-fit',zorder=1,linewidth=2.5)
axs1.plot(k, dP_dk_ca(k)*(k/power_spectrum_LCDM_alldata), color = 'black', label=r'All data LambdaCDM best-fit',zorder=3, linewidth=2.5)
axs1.set_xscale('log')
axs1.set_xlim([5e-1,2])
axs1.set_ylim([-2.5,-2])
axs1.legend(fontsize='12',ncol=1,loc='upper right')
axs1.set_xlabel('k \ [h/Mpc]',fontsize=21, labelpad=12)
axs1.set_ylabel(r'dlnP / dlnk', fontsize=21, labelpad=12)

axs1.errorbar(k[index], -2.34, yerr=0.012,color = 'black',marker = '_',capsize=2,zorder=5)

fig1.tight_layout()
fig1.savefig('derivs.pdf')
