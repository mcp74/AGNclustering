import numpy as np 
from numpy.lib.recfunctions import append_fields
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
from astropy import units as u
import sys
import random
import math

#import genrand
from Corrfunc.mocks import DDrppi_mocks
from AGNclustering.utils import *


def sum_rp(xi,rpbins,pibins):
	wppi = np.zeros(len(pibins))
	k=0
	for i in range(len(rpbins)-1):     
		for j in range(len(pibins)):
			wppi[j] = wppi[j] + 2.0*xi[j+k]
		k+=len(pibins)
	return wppi


def wppi_auto_xi(nd,nr,pibins,rpbins,pimax,dd,dr,rr=None,estimator='L'):
	nb = len(dd)
	nbins = len(rpbins)-1
#     Rebin pi in terms of pibins
	rebinnedlength = len(pibins)*len(dd)/pimax
#     Define rebinned npairs column
	ddtemp=np.zeros(int(rebinnedlength))
	drtemp=np.zeros(int(rebinnedlength))
	rrtemp=np.zeros(int(rebinnedlength))
	npibins = len(pibins)
#     Loop through data and fill in rebinned npair column
	for j in range(len(rpbins)-1):
		for i in range(len(pibins)):
			if i==0:
				ddtemp[0+(npibins*j)] = np.sum(dd['npairs'][0+j*pimax:pibins[0]+j*pimax])
				drtemp[0+(npibins*j)] = np.sum(dd['npairs'][0+j*pimax:pibins[0]+j*pimax])
				if (estimator=='L') or (estimator=='Landy'):
					rrtemp[0+(npibins*j)] = np.sum(rr['npairs'][0+j*pimax:pibins[0]+j*pimax])
			else:
				ddtemp[i+(npibins*j)] = np.sum(dd['npairs'][pibins[i-1]+j*pimax:pibins[i]+j*pimax])
				drtemp[i+(npibins*j)] = np.sum(dd['npairs'][pibins[i-1]+j*pimax:pibins[i]+j*pimax])
				if (estimator=='L') or (estimator=='Landy'):
					rrtemp[i+(npibins*j)] = np.sum(rr['npairs'][pibins[i-1]+j*pimax:pibins[i]+j*pimax])
                
	DD = ddtemp/(nd*(nd-1.))
	DR = drtemp/(nd*nr)

	if (estimator=='L') or (estimator=='Landy'):
		RR = rrtemp/(nr*(nr-1.))
		nonzero = RR > 0
		xi = np.zeros(int(rebinnedlength))
		xi[nonzero] = (DD[nonzero] - 2 * DR[nonzero] + RR[nonzero]) / RR[nonzero]
	elif (estimator=='P') or (estimator=='Peebles'):
		nonzero = DR > 0
		xi = np.zeros(int(rebinnedlength))
		xi[nonzero] = (DD[nonzero] / DR[nonzero] ) - 1.
	elif (estimator=='H') or (estimator=='Hamilton'):
		RR = rr['npairs']/(nr*(nr-1.))
		nonzero = DR > 0
		xi = np.zeros(int(rebinnedlength))
		xi[nonzero] = (DD[nonzero] * RR[nonzero] / (DR[nonzero] * DR[nonzero]) ) - 1.
	else:
		sys.exit('Not a valid estimator')
	return xi


def wppi_cross_xi(nd1,nd2,nr1,nr2,pibins,rpbins,pimax,d1d2,d1r2,d2r1=None,r1r2=None,estimator='L'):
	nb = len(d1d2)
	nbins = len(rpbins)-1
#     Rebin pi in terms of pibins
	rebinnedlength = len(pibins)*len(d1d2)/pimax
#     Define rebinned npairs column
	d1d2temp=np.zeros(int(rebinnedlength))
	d1r2temp=np.zeros(int(rebinnedlength))
	d2r1temp=np.zeros(int(rebinnedlength))
	r1r2temp=np.zeros(int(rebinnedlength))
# Define length of pi bins
	npibins = len(pibins)
#     Loop through data and fill in rebinned npair column
	for j in range(nbins):
		for i in range(len(pibins)):
			if i==0:
				d1d2temp[0+(npibins*j)] = np.sum(d1d2['npairs'][0+j*pimax:pibins[0]+j*pimax])
				d1r2temp[0+(npibins*j)] = np.sum(d1r2['npairs'][0+j*pimax:pibins[0]+j*pimax])
				if (estimator=='L') or (estimator=='Landy'):
					d2r1temp[0+(npibins*j)] = np.sum(d2r1['npairs'][0+j*pimax:pibins[0]+j*pimax])
					r1r2temp[0+(npibins*j)] = np.sum(r1r2['npairs'][0+j*pimax:pibins[0]+j*pimax])
			else:
				d1d2temp[i+(npibins*j)] = np.sum(d1d2['npairs'][pibins[i-1]+j*pimax:pibins[i]+j*pimax])
				d1r2temp[i+(npibins*j)] = np.sum(d1r2['npairs'][pibins[i-1]+j*pimax:pibins[i]+j*pimax])
				if (estimator=='L') or (estimator=='Landy'):
					d2r1temp[i+(npibins*j)] = np.sum(d2r1['npairs'][pibins[i-1]+j*pimax:pibins[i]+j*pimax])
					r1r2temp[i+(npibins*j)] = np.sum(r1r2['npairs'][pibins[i-1]+j*pimax:pibins[i]+j*pimax])

#     Redefine D1D2 and D1R2array-like or list of colors or color, option
	D1D2 = d1d2temp/(nd1*nd2)
	D1R2 = d1r2temp/(nd1*nr2)

	#RR = rr['npairs']/(nr*(nr-1.))

	if (estimator=='L') or (estimator=='Landy'):
		D2R1 = d2r1temp/(nd2*nr1)
		R1R2 = r1r2temp/(nr1*nr2)

		nonzero = R1R2 > 0
		xi = np.zeros(int(rebinnedlength))
		xi[nonzero] = (D1D2[nonzero] - D1R2[nonzero] - D2R1[nonzero] + R1R2[nonzero]) / R1R2[nonzero]

	elif (estimator=='P') or (estimator=='Peebles'):
		nonzero = D1R2 > 0
		xi = np.zeros(int(rebinnedlength))
		xi[nonzero] = (D1D2[nonzero] / D1R2[nonzero] ) - 1.
	#elif (estimator=='H') or (estimator=='Hamilton'):
	#    nonzero = DR > 0
	#    xi = np.zeros(nb)
	#    xi[nonzero] = (DD[nonzero] * RR[nonzero] / (DR[nonzero] * DR[nonzero]) ) - 1.
	else:
		sys.exit('Not a valid estimator')
	return xi

def wppi_dd(data, randoms, pibins, rpbins, pimax, estimator='L',weights=None):

	if 'weight' in data.dtype.names:
		weights=data['weight']
		rweights = np.ones(len(randoms))

	if weights is not None:
		dd = weighted_pair_count(rpbins,pimax,data['ra'],data['dec'],data['cdist'],ra2=None,dec2=None,cd2=None,weights1=weights)
		dr = weighted_pair_count(rpbins,pimax,data['ra'],data['dec'],data['cdist'],ra2=randoms['ra'],dec2=randoms['dec'],cd2=randoms['cdist'],weights1=weights,weights2=rweights)
		rr = weighted_pair_count(rpbins,pimax,randoms['ra'],randoms['dec'],randoms['cdist'],ra2=None,dec2=None,cd2=None,weights1=rweights)
	else:
		dd = pair_count(rpbins,pimax,data['ra'],data['dec'],data['cdist'],ra2=None,dec2=None,cd2=None)
		dr = pair_count(rpbins,pimax,data['ra'],data['dec'],data['cdist'],ra2=randoms['ra'],dec2=randoms['dec'],cd2=randoms['cdist'])
		rr = pair_count(rpbins,pimax,randoms['ra'],randoms['dec'],randoms['cdist'],ra2=None,dec2=None,cd2=None)

	nd = len(data)
	nr = len(randoms)
	if weights is not None:
		xit = weighted_autoxi(np.sum(weights),nr,rpbins,dd,dr,rr,estimator=estimator)
	else:
		xit = wppi_auto_xi(nd,nr,pibins,rpbins,pimax,dd,dr,rr,estimator=estimator)
	wppi = sum_rp(xit,rpbins,pibins)

	return wppi

def wppi_d1d2(d1, d2, r2, pibins, rpbins, pimax, r1=None, estimator='L',weights1=None,weights2=None):

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
		d1d2 = weighted_pair_count(rpbins,pimax,d1['ra'],d1['dec'],d1['cdist'],ra2=d2['ra'],dec2=d2['dec'],cd2=d2['cdist'],weights1=weights1,weights2=weights2)
		d1r2 = weighted_pair_count(rpbins,pimax,d1['ra'],d1['dec'],d1['cdist'],ra2=r2['ra'],dec2=r2['dec'],cd2=r2['cdist'],weights1=weights1,weights2=np.ones(len(r2)))

	else:
		d1d2 = pair_count(rpbins,pimax,d1['ra'],d1['dec'],d1['cdist'],ra2=d2['ra'],dec2=d2['dec'],cd2=d2['cdist'])
		d1r2 = pair_count(rpbins,pimax,d1['ra'],d1['dec'],d1['cdist'],ra2=r2['ra'],dec2=r2['dec'],cd2=r2['cdist'])
	
	if (estimator=='L') or (estimator=='Landy'):
		if r1 is None:
			sys.exit('Need random catalog for d1 !')
		if weights2 is not None:
			d2r1 = weighted_pair_count(rpbins,pimax,d2['ra'],d2['dec'],d2['cdist'],ra2=r1['ra'],dec2=r1['dec'],cd2=r1['cdist'],weights1=weights2,weights2=np.ones(len(r1)))
			r1r2 = weighted_pair_count(rpbins,pimax,r1['ra'],r1['dec'],r1['cdist'],ra2=r2['ra'],dec2=r2['dec'],cd2=r2['cdist'],weights1=np.ones(len(r1)),weights2=np.ones(len(r2)))
		else:
			d2r1 = pair_count(rpbins,pimax,d2['ra'],d2['dec'],d2['cdist'],ra2=r1['ra'],dec2=r1['dec'],cd2=r1['cdist'])
			r1r2 = pair_count(rpbins,pimax,r1['ra'],r1['dec'],r1['cdist'],ra2=r2['ra'],dec2=r2['dec'],cd2=r2['cdist'])
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
		xit = weighted_cross_xi(np.sum(weights1),np.sum(weights2),nr1,nr2,rpbins,d1d2,d1r2,d2r1,r1r2,estimator=estimator)
	else:
		if pimax==1.0:
			xit=cross_xi(nd1,nd2,nr1,nr2,rpbins,d1d2,d1r2,d2r1,r1r2,estimator=estimator)
		else:
			xit = wppi_cross_xi(nd1,nd2,nr1,nr2,pibins,rpbins,pimax,d1d2,d1r2,d2r1,r1r2,estimator=estimator)

	wppi = sum_rp(xit,rpbins,pibins)
    
	return wppi

