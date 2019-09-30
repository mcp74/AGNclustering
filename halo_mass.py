import numpy as np
import math
from scipy.interpolate import interp1d
import scipy.integrate as intg
from astropy.cosmology import FlatLambdaCDM,FLRW,Planck15

from hmf import MassFunction


delta_c = 1.686

def bias_T10(nu):
	'''
	Tinker+2010 bias function
	'''
    delta_halo = 200
    y = np.log10(delta_halo)
    
    A = 1.0 + 0.24 * y * np.exp(-(4 / y) ** 4)
    a = 0.44 * y - 0.88
    C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4 / y) ** 4)
    #nu = np.sqrt(self.nu)
    B = .183
    c = 2.4
    b = 1.5
    return 1 - A * nu ** a / (nu ** a + delta_c ** a) + B * nu ** b + C * nu ** c


def bias_T05(nu):
	'''
	Tinker+2005 bias function
	'''
    a= .707
    b=.35
    c=.8
    #print(nu,delta(z))
    return 1+(1./(np.sqrt(a)*delta_c)) * ( (np.sqrt(a)*a*nu**2) + np.sqrt(a)*b*(a*nu**2)**(1-c) - \
    	((a*nu**2)**c/((a*nu**2)**c + b*(1-c)*(1-c/2.))) )


def bias_S01(nu):
	'''
	Sheth+2001 bias function
	'''
    a= .707
    b=.5
    c=.6
    return 1+(1./(np.sqrt(a)*delta_c)) * ( (np.sqrt(a)*a*nu**2) + np.sqrt(a)*b*(a*nu**2)**(1-c) - \
    	((a*nu**2)**c/((a*nu**2)**c + b*(1-c)*(1-c/2.))) )

def m_to_nu(M,z,cosmo):
    hmf = MassFunction(Mmin=9,Mmax=16,z=z,cosmo_model=Planck15)
    return delta_c/sigma(M,z,cosmo)

def P_k(k,z,cosmo):
	'''
	DM linear power spectrum, using transfer function from Eisenstein & Hu (1998) from hmf
	'''
    hmf = MassFunction(Mmin=9,Mmax=16,z=z,cosmo_model=cosmo,transfer_model='EH_BAO')
    f = interp1d(hmf.k,hmf.power,kind='cubic')
    return f(k)

def mean_density0(cosmo):
    return (cosmo.Om0 * cosmo.critical_density0 / (cosmo.h**2)).to(u.Msun/u.Mpc**3).value

def radii(mass,cosmo):
    return (3.*mass / (4.*np.pi * mean_density0(cosmo))) ** (1. / 3.)

def k_space(kr):
    return np.where(kr>1.4e-6,(3 / kr ** 3) * (np.sin(kr) - kr * np.cos(kr)),1)   


def sigma(mass,z,cosmo):
	'''
	root-mean square of mass density fluctuations withina sphere containing mass M
	'''
    lnk_min=np.log(1e-8)
    lnk_max=np.log(2e4)
    dlnk=0.05
    k_arr = np.exp(np.arange(lnk_min, lnk_max, dlnk))

    r = radii(mass,cosmo)
    rk = np.outer(r,k_arr)

    # we multiply by k because our steps are in logk.
    rest = P_k(k_arr,z,cosmo) * k_arr**3 
    integ = rest*k_space(rk)**2
    sig = (0.5/np.pi**2) * intg.simps(integ,dx=dlnk,axis=-1)
    return np.sqrt(sig)

def Mh(b,z,cosmo,bias_model='T10'):
	'''
	returns effective halo mass from bias and redshift.
	Default bias model is Tinker+2010 ('T10'). Other options are Tinker+2005 ('T05') and Sheth & Tormen 2001 ('S01')
	'''
    m_arr = 10 ** np.arange(9, 16, 0.01)
    nu_arr = m_to_nu(m_arr,z,cosmo)
    if bias_model=='T10':
    	f=interp1d(bias_T10(nu_arr),m_arr,kind='cubic')
    elif bias_model=='T05':
    	f=interp1d(bias_T05(nu_arr),m_arr,kind='cubic')
    elif bias_model=='S01':
    	f=interp1d(bias_S01(nu_arr),m_arr,kind='cubic')
    return f(b)

