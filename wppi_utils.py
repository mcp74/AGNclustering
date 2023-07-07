import numpy as np 
from numpy.lib.recfunctions import append_fields
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
from astropy import units as u
import sys

#import genrand
from Corrfunc.mocks import DDrppi_mocks
from AGNclustering.utils import *

# New function to sum along rp
def sum_rp(xi,pibins):
	wppi = np.zeros(40)
	k=0
	for i in range(10):     
		for j in range(40):
			wppi[j] = wppi[j] + 2.0*xi[j+k]
		k+=40
	return wppi

def wppi_cross_xi(nd1,nd2,nr1,nr2,bins,pibins,pimax,d1d2,d1r2,d2r1=None,r1r2=None,estimator='L'):
	nb = len(d1d2)
	nbins = len(bins)-1
#     Rebin pi in terms of pibins
	templength = len(pibins)*len(d1d2)/pimax
	d1d2temp=np.zeros(int(templength))
	print(len(d1d2temp))
	for i in range(len(pibins)-1):
		if i==0:
			d1d2temp[0] = np.sum(d1d2['pimax'][0:pibins[0]])
		else:
			d1d2temp[i] = np.sum(d1d2['pimax'][pibins[i-1]:pibins[i]])                                              
	print(d1d2temp)
	
	D1D2 = d1d2['npairs']/(nd1*nd2)
	D1R2 = d1r2['npairs']/(nd1*nr2)


	#RR = rr['npairs']/(nr*(nr-1.))

	if (estimator=='L') or (estimator=='Landy'):
		D2R1 = d2r1['npairs']/(nd2*nr1)
		R1R2 = r1r2['npairs']/(nr1*nr2)

		nonzero = R1R2 > 0
		xi = np.zeros(nb)
		xi[nonzero] = (D1D2[nonzero] - D1R2[nonzero] - D2R1[nonzero] + R1R2[nonzero]) / R1R2[nonzero]

	elif (estimator=='P') or (estimator=='Peebles'):
		nonzero = D1R2 > 0
		xi = np.zeros(nb)
		xi[nonzero] = (D1D2[nonzero] / D1R2[nonzero] ) - 1.
	#elif (estimator=='H') or (estimator=='Hamilton'):
	#    nonzero = DR > 0
	#    xi = np.zeros(nb)
	#    xi[nonzero] = (DD[nonzero] * RR[nonzero] / (DR[nonzero] * DR[nonzero]) ) - 1.
	else:
		sys.exit('Not a valid estimator')
	return xi


def wppi_d1d2(d1, d2, r2, bins, pimax, pibins, r1=None, estimator='L',weights1=None,weights2=None):

	if ('weight' in d1.dtype.names):
		weights1=d1['weight']
	if ('weight' in d2.dtype.names):
		weights2=d2['weight']

	if (weights1 is not None) or (weights2 is not None):
		if weights1 is None:
			weights1 = np.ones(len(d1))
		if weights2 is None:
			weights2 = np.ones(len(d2))
		#print(d1['ra'],d1['dec'],d1['cdist'],d2['ra'],d2['dec'],d2['cdist'],weights1,weights2)
		d1d2 = weighted_pair_count(bins,pimax,d1['ra'],d1['dec'],d1['cdist'],ra2=d2['ra'],dec2=d2['dec'],cd2=d2['cdist'],weights1=weights1,weights2=weights2)
		d1r2 = weighted_pair_count(bins,pimax,d1['ra'],d1['dec'],d1['cdist'],ra2=r2['ra'],dec2=r2['dec'],cd2=r2['cdist'],weights1=weights1,weights2=np.ones(len(r2)))

	else:
		d1d2 = pair_count(bins,pimax,d1['ra'],d1['dec'],d1['cdist'],ra2=d2['ra'],dec2=d2['dec'],cd2=d2['cdist'])
		d1r2 = pair_count(bins,pimax,d1['ra'],d1['dec'],d1['cdist'],ra2=r2['ra'],dec2=r2['dec'],cd2=r2['cdist'])
	
	if (estimator=='L') or (estimator=='Landy'):
		if r1 is None:
			sys.exit('Need random catalog for d1 !')
		if weights2 is not None:
			d2r1 = weighted_pair_count(bins,pimax,d2['ra'],d2['dec'],d2['cdist'],ra2=r1['ra'],dec2=r1['dec'],cd2=r1['cdist'],weights1=weights2,weights2=np.ones(len(r1)))
			r1r2 = weighted_pair_count(bins,pimax,r1['ra'],r1['dec'],r1['cdist'],ra2=r2['ra'],dec2=r2['dec'],cd2=r2['cdist'],weights1=np.ones(len(r1)),weights2=np.ones(len(r2)))
		else:
			d2r1 = pair_count(bins,pimax,d2['ra'],d2['dec'],d2['cdist'],ra2=r1['ra'],dec2=r1['dec'],cd2=r1['cdist'])
			r1r2 = pair_count(bins,pimax,r1['ra'],r1['dec'],r1['cdist'],ra2=r2['ra'],dec2=r2['dec'],cd2=r2['cdist'])
	else: 
		d2r1 = None
		r1r2 = None

	nd1 = len(d1)
	nd2 = len(d2)
	if r1 is None:
		nr1=0
	else: nr1 = len(r1)
	nr2 = len(r2)
	if (weights1 is not None) or (weights2 is not None):
		xit = weighted_cross_xi(np.sum(weights1),np.sum(weights2),nr1,nr2,bins,d1d2,d1r2,d2r1,r1r2,estimator=estimator)
	else:
		xit = wppi_cross_xi(nd1,nd2,nr1,nr2,bins,pibins,pimax,d1d2,d1r2,d2r1,r1r2,estimator=estimator)

# 	wp = sum_pi(xit,bins)
    
#     Calling the new sum rp function in zp_utils
	wppi = sum_rp(xit,pibins)
    
	return wppi