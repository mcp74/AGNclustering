from __future__ import division

import numpy as np 
from numpy.lib.recfunctions import append_fields
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
from astropy import units as u
import sys

#import genrand
from Corrfunc.mocks import DDrppi_mocks


def pair_count(rpbins,pimax,ra1,dec1,cd1,ra2=None,dec2=None,cd2=None,nthreads=2):
	d1ra = ra1.astype('float64')
	d1dec = dec1.astype('float64')
	d1cd = cd1.astype('float64')

	if ra2 is None:
		nn = DDrppi_mocks(autocorr=1, cosmology=1, nthreads=nthreads, pimax=pimax, binfile=rpbins, RA1=d1ra, DEC1=d1dec,\
						  CZ1=d1cd, is_comoving_dist=True, output_rpavg=True, verbose=True)

	else:
		d2ra = ra2.astype('float64')
		d2dec = dec2.astype('float64')
		d2cd = cd2.astype('float64')
		nn = DDrppi_mocks(autocorr=0, cosmology=1, nthreads=nthreads, pimax=pimax, binfile=rpbins, RA1=d1ra, DEC1=d1dec, \
			  CZ1=d1cd, RA2=d2ra, DEC2=d2dec, CZ2=d2cd, is_comoving_dist=True, \
			  output_rpavg=True, verbose=True)        
		
	return nn


def weighted_pair_count(rpbins,pimax,ra1,dec1,cd1,weights1=None,ra2=None,dec2=None,cd2=None,weights2=None,nthreads=2):
	d1ra = ra1.astype('float64')
	d1dec = dec1.astype('float64')
	d1cd = cd1.astype('float64')

	if ra2 is None:
		nn = DDrppi_mocks(autocorr=1, cosmology=1, nthreads=nthreads, pimax=pimax, binfile=rpbins, RA1=d1ra, DEC1=d1dec,\
						  CZ1=d1cd, weights1=weights1, is_comoving_dist=True, output_rpavg=True, verbose=True, weight_type='pair_product')

	else:
		d2ra = ra2.astype('float64')
		d2dec = dec2.astype('float64')
		d2cd = cd2.astype('float64')
		nn = DDrppi_mocks(autocorr=0, cosmology=1, nthreads=nthreads, pimax=pimax, binfile=rpbins, RA1=d1ra, DEC1=d1dec, \
			  CZ1=d1cd, weights1=weights1, RA2=d2ra, DEC2=d2dec, CZ2=d2cd, weights2=weights2, is_comoving_dist=True, \
			  output_rpavg=True, verbose=True, weight_type='pair_product')        
		
	return nn


def auto_xi(nd,nr,bins,dd,dr,rr=None,estimator='L'):
	nb = len(dd)
	nbins = len(bins)-1

	DD = dd['npairs']/(nd*(nd-1.))
	DR = dr['npairs']/(nd*nr)

	if (estimator=='L') or (estimator=='Landy'):
		RR = rr['npairs']/(nr*(nr-1.))
		nonzero = RR > 0
		xi = np.zeros(nb)
		xi[nonzero] = (DD[nonzero] - 2 * DR[nonzero] + RR[nonzero]) / RR[nonzero]
	elif (estimator=='P') or (estimator=='Peebles'):
		nonzero = DR > 0
		xi = np.zeros(nb)
		xi[nonzero] = (DD[nonzero] / DR[nonzero] ) - 1.
	elif (estimator=='H') or (estimator=='Hamilton'):
		RR = rr['npairs']/(nr*(nr-1.))
		nonzero = DR > 0
		xi = np.zeros(nb)
		xi[nonzero] = (DD[nonzero] * RR[nonzero] / (DR[nonzero] * DR[nonzero]) ) - 1.
	else:
		sys.exit('Not a valid estimator')
	return xi


def weighted_autoxi(nd,nr,nbins,dd,dr,rr,estimator='Hamilton'):
	nb = len(dd)

	wdd = dd['weightavg']/np.average(dd['weightavg'])
	wdr = dr['weightavg']/np.average(dr['weightavg'])
	wrr = rr['weightavg']/np.average(rr['weightavg'])

	DD = dd['npairs']*wdd/(nd*(nd-1.))
	DR = dr['npairs']*wdr/(nd*nr)
	RR = rr['npairs']*wrr/(nr*(nr-1.))

	if (estimator=='L') or (estimator=='Landy'):
		nonzero = RR > 0
		xi = np.zeros(nb)
		xi[nonzero] = (DD[nonzero] - 2 * DR[nonzero] + RR[nonzero]) / RR[nonzero]
	elif (estimator=='P') or (estimator=='Peebles'):
		nonzero = DR > 0
		xi = np.zeros(nb)
		xi[nonzero] = (DD[nonzero] / DR[nonzero] ) - 1.
	elif (estimator=='H') or (estimator=='Hamilton'):
		nonzero = DR > 0
		xi = np.zeros(nb)
		xi[nonzero] = (DD[nonzero] * RR[nonzero] / (DR[nonzero] * DR[nonzero]) ) - 1.
	else:
		sys.exit('Not a valid estimator')
	return xi


def cross_xi(nd1,nd2,nr1,nr2,bins,d1d2,d1r2,d2r1=None,r1r2=None,estimator='L'):
	nb = len(d1d2)
	nbins = len(bins)-1
	
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
		nonzero = DR > 0
		xi = np.zeros(nb)
		xi[nonzero] = (D1D2[nonzero] / D1R2[nonzero] ) - 1.
	#elif (estimator=='H') or (estimator=='Hamilton'):
	#    nonzero = DR > 0
	#    xi = np.zeros(nb)
	#    xi[nonzero] = (DD[nonzero] * RR[nonzero] / (DR[nonzero] * DR[nonzero]) ) - 1.
	else:
		sys.exit('Not a valid estimator')
	return xi


def sum_pi(xi,rpbins):
	nbins = len(rpbins)-1
	npibins = len(xi) // nbins
	dpi = 1
	wp = np.empty(nbins)

	for i in range(nbins):
		wp[i] = 2.0 * dpi * np.sum(xi[i * npibins:(i + 1) * npibins])

	return wp


def wp_dd(data, randoms, bins, pimax, estimator='L'):

	dd = pair_count(bins,pimax,data['ra'],data['dec'],data['cdist'],ra2=None,dec2=None,cd2=None)
	dr = pair_count(bins,pimax,data['ra'],data['dec'],data['cdist'],ra2=randoms['ra'],dec2=randoms['dec'],cd2=randoms['cdist'])
	rr = pair_count(bins,pimax,randoms['ra'],randoms['dec'],randoms['cdist'],ra2=None,dec2=None,cd2=None)

	nd = len(data)
	nr = len(randoms)
	xit = auto_xi(nd,nr,bins,dd,dr,rr,estimator=estimator)
	wp = sum_pi(xit,bins)

	return wp


def wp_d1d2(d1, d2, r2, bins, pimax, r1=None, estimator='L'):

	d1d2 = pair_count(bins,pimax,d1['ra'],d1['dec'],d1['cdist'],ra2=d2['ra'],dec2=d2['dec'],cd2=d2['cdist'])
	d1r2 = pair_count(bins,pimax,d1['ra'],d1['dec'],d1['cdist'],ra2=r2['ra'],dec2=r2['dec'],cd2=r2['cdist'])
	
	if (estimator=='L') or (estimator=='Landy'):
		if r1 is None:
			sys.exit('Need random catalog for d1 !')
		d2r1 = pair_count(bins,pimax,d2['ra'],d2['dec'],d2['cdist'],ra2=r1['ra'],dec2=r1['dec'],cd2=r1['cdist'])
		r1r2 = pair_count(bins,pimax,r1['ra'],r1['dec'],r1['cdist'],ra2=r2['ra'],dec2=r2['dec'],cd2=r2['cdist'])

	nd1 = len(d1)
	nd2 = len(d2)
	if r1 is None:
		nr1=0
	else: nr1 = len(r1)
	nr2 = len(r2)
	xit = cross_xi(nd1,nd2,nr1,nr2,bins,d1d2,d1r2,d2r1,r1r2,estimator='L')
	wp = sum_pi(xit,bins)

	return wp


