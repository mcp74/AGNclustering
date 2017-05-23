import numpy as np
import math
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM,Planck15
from scipy.integrate import quad,romberg
from scipy.interpolate import interp1d
from scipy.special import j0
import 

from halomod import HaloModel, bias
from hmf import MassFunction

__author__ = "Meredith Powell <meredith.powell@yale.edu>"

class HOD(object):

	def __init__(self,Mcmin,Msmin,M1,alpha):
		'''
		Initialize halo mass profile (NFW), halo mass function (Tinker 08), halo bias (SMT 01)
		'''
		#HOD params:
		self.Mcmin=Mcmin
		self.Msmin=Msmin
		self.M1=M1
		self.alpha=alpha

		#integration bounds:
		self.log_Mhmin = 9
		self.log_Mhmax = 16
		#default models:
		self.nfw_model='NFW'
		self.hmf_model='Tinker_08'
		self.hbias_model='SMT_01'

		self.hm = HaloModel(profile_model="NFW")
		self.u_nfw = self.hm.profile.u
		self.Plin = interp1d(self.hm.k,self.hm.power,kind='cubic')

		hmf = MassFunction(Mmin=self.log_Mhmin,Mmax=self.log_Mhmax+.5)
		self.phi = interp1d(hmf.m,hmf.dndlog10m, kind='cubic')

		SMTbias = bias.SMT01(self.hm.nu**2,delta_c=hmf.delta_c)
		bh = SMTbias.bias()
		self.m_to_nu = interp1d(np.log10(hmf.m),np.sqrt(hmf.nu), kind='cubic')
		self.nu_to_hbias = interp1d(np.sqrt(SMTbias.nu),bh, kind='cubic')
		self._calculate_n()
		self._calculate_bias()

	def bias_h(logm_arr):
    	nu_arr=m_to_nu(logm_arr)
    	return nu_to_hbias(nu_arr)


	def Nsat(self,Mh):
		if Mh>self.Msmin:
			return (Mh/self.M1)**self.alpha
		else:
			return 0

	def Ncen(self,Mh):
		if Mh>self.Mcmin:
			return 1
		else:
			return 0

	def Ntot(self,Mh):
		return self.Nsat(Mh) + self.Ncen(Mh)

	def _b_integrand(log_Mh):
		return self.bias_h(log_Mh)*self.Ntot(log_Mh)*self.phi(10**log_Mh)

	def _n_integrand(log_Mh):
		return self.Ntot(10**log_Mh)*self.phi(10**log_Mh)

	def _calculate_bias(self):
		num = quad(_b_integrand,self.log_Mhmin,self.log_Mhmax)[0]
		den = self.n
		self.bias = num/den

	def _calculate_n(self):
		self.n = quad(_n_integrand,self.log_Mhmin,self.log_Mhmax)[0]

	def _integrand_p1h(log_Mh,k):
		Mh = 10**log_Mh
		Ns = self.Nsat(Mh)
    	Nc = self.Ncen(Mh)
    	y = self.u_nfw(k,Mh)
		return self.phi(Mh)*( Nc*Ns * y + Ns*(Ns-1)*y**2)

    def _wpdmintegrand(log_k,r):
    	k = 10**log_k
    	dk = k * np.log(10)
    	return (dk * k/(2.*np.pi)) * Plin(k) * j0(k * r) 

	def _wp1integrand(log_k,r):
    	k = 10**log_k
    	dk = k * np.log(10)
    	return (dk * k/(2.*np.pi)) * P1h(k) * j0(k * r) 


	def _wp2integrand(log_k,r):
    	k = 10**log_k
    	dk = k * np.log(10)
    	return (dk * k / (2. * np.pi) * P2h(k) * j0(k * r))

    def P1h(self,k):
		x = quad(_integrand_p1h,self.log_Mhmin,self.log_Mhmax,args=(k))[0]
		return self.n**-2*(2*np.pi)**-3*x

	def P2h(self,k):
    	return self.bias**2*self.Plin(k)

    def wpdm(self,rp):
    	return quad(_wpdmintegrand,-2,2,args=(rp))[0]

	def wp2h(self,rp):
    	return quad(_wp2integrand,-2,2,args=(rp))[0]

	def wp1h(self,rp):
    	return quad(_wp2integrand,-2,2,args=(rp))[0]

    def wp(self,rp):
    	return wp1h(rp)+wp2h(rp)

