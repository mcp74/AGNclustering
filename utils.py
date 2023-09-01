import numpy as np 
from numpy.lib.recfunctions import append_fields
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
from astropy import units as u
import sys

#import genrand
from Corrfunc.mocks import DDrppi_mocks
from Corrfunc.mocks import DDsmu_mocks

def pair_count(rpbins,pimax,ra1,dec1,cd1,ra2=None,dec2=None,cd2=None,nthreads=4,mubins=None):
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
		if mubins is not None:
			nmu_bins=len(mubins)-1
			nn = DDsmu_mocks(autocorr=0, cosmology=1, nthreads=nthreads, mu_max=pimax, nmu_bins=nmu_bins, binfile=rpbins, RA1=d1ra, DEC1=d1dec, CZ1=d1cd, RA2=d2ra, DEC2=d2dec, CZ2=d2cd, is_comoving_dist=True, output_savg=True, verbose=True)
		else:
			nn = DDrppi_mocks(autocorr=0, cosmology=1, nthreads=nthreads, pimax=pimax, binfile=rpbins, RA1=d1ra, DEC1=d1dec, \
						CZ1=d1cd, RA2=d2ra, DEC2=d2dec, CZ2=d2cd, is_comoving_dist=True, \
						output_rpavg=True, verbose=True)   
	return nn


def weighted_pair_count(rpbins,pimax,ra1,dec1,cd1,weights1,weights2=None,ra2=None,dec2=None,cd2=None,nthreads=4):
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


def weighted_autoxi(nd,nr,nbins,dd,dr,rr,estimator='P'):
	nb = len(dd)

	wdd = dd['weightavg']
	wdr = dr['weightavg']
	DD = dd['npairs']*wdd/(nd*(nd-1.))
	DR = dr['npairs']*wdr/(nd*nr)

	if (estimator=='L') or (estimator=='Landy'):
		wrr = rr['weightavg']
		RR = rr['npairs']*wrr/(nr*(nr-1.))
		nonzero = RR > 0
		xi = np.zeros(nb)
		xi[nonzero] = (DD[nonzero] - 2 * DR[nonzero] + RR[nonzero]) / RR[nonzero]
	elif (estimator=='P') or (estimator=='Peebles'):
		nonzero = DR > 0
		xi = np.zeros(nb)
		xi[nonzero] = (DD[nonzero] / DR[nonzero] ) - 1.
	elif (estimator=='H') or (estimator=='Hamilton'):
		wrr = rr['weightavg']
		RR = rr['npairs']*wrr/(nr*(nr-1.))
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

def weighted_cross_xi(nd1,nd2,nr1,nr2,bins,d1d2,d1r2,d2r1=None,r1r2=None,estimator='L'):
	nb = len(d1d2)
	nbins = len(bins)-1
	#npibins = len(d1d2) // nbins
	#(npibins)
	
	wd1d2 = d1d2['weightavg']
	wd1r2 = d1r2['weightavg']
	D1D2 = d1d2['npairs']*wd1d2/(nd1*nd2)
	D1R2 = d1r2['npairs']*wd1r2/(nd1*nr2)

	if (estimator=='L') or (estimator=='Landy'):
		wd2r1 = d2r1['weightavg']
		wr1r2 = r1r2['weightavg']
		D2R1 = d2r1['npairs']*wd2r1/(nd2*nr1)
		R1R2 = r1r2['npairs']*wr1r2/(nr1*nr2)

		nonzero = R1R2 > 0
		xi = np.zeros(nb)
		xi[nonzero] = (D1D2[nonzero] - D1R2[nonzero] - D2R1[nonzero] + R1R2[nonzero]) / R1R2[nonzero]

	elif (estimator=='P') or (estimator=='Peebles'):
		nonzero = D1R2 > 0
		xi = np.zeros(nb)
		xi[nonzero] = (D1D2[nonzero] / D1R2[nonzero] ) - 1.

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

def wp_dd(data, randoms, bins, pimax, estimator='L',weights=None):

	if 'weight' in data.dtype.names:
		weights=data['weight']
		rweights = np.ones(len(randoms))

	if weights is not None:
		dd = weighted_pair_count(bins,pimax,data['ra'],data['dec'],data['cdist'],ra2=None,dec2=None,cd2=None,weights1=weights)
		dr = weighted_pair_count(bins,pimax,data['ra'],data['dec'],data['cdist'],ra2=randoms['ra'],dec2=randoms['dec'],cd2=randoms['cdist'],weights1=weights,weights2=rweights)
		rr = weighted_pair_count(bins,pimax,randoms['ra'],randoms['dec'],randoms['cdist'],ra2=None,dec2=None,cd2=None,weights1=rweights)
	else:
		dd = pair_count(bins,pimax,data['ra'],data['dec'],data['cdist'],ra2=None,dec2=None,cd2=None)
		dr = pair_count(bins,pimax,data['ra'],data['dec'],data['cdist'],ra2=randoms['ra'],dec2=randoms['dec'],cd2=randoms['cdist'])
		rr = pair_count(bins,pimax,randoms['ra'],randoms['dec'],randoms['cdist'],ra2=None,dec2=None,cd2=None)

	nd = len(data)
	nr = len(randoms)
	if weights is not None:
		xit = weighted_autoxi(np.sum(weights),nr,bins,dd,dr,rr,estimator=estimator)
	else:
		xit = auto_xi(nd,nr,bins,dd,dr,rr,estimator=estimator)
	wp = sum_pi(xit,bins)

	return wp


def wp_d1d2(d1, d2, r2, bins, pimax, r1=None, estimator='L',weights1=None,weights2=None):

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
		xit = cross_xi(nd1,nd2,nr1,nr2,bins,d1d2,d1r2,d2r1,r1r2,estimator=estimator)

	wp = sum_pi(xit,bins)
    
    
	return wp

def z_to_cdist(data,cosmo=None):
	if cosmo is None:
		cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
		print('using LCDM cosmology: Om0=.3, H0=70')
	cdists = np.array([cosmo.comoving_distance(z).value for z in data['z']])*cosmo.h
	data = append_fields(data, 'cdist', cdists)
	return np.array(data)


