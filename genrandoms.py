import numpy as np 
import random
from scipy.stats import gaussian_kde
from astropy.table import Table
import matplotlib.pyplot as plt
import sys
from numpy.lib.recfunctions import append_fields
from astropy import units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord, match_coordinates_sky
from kde import weighted_gaussian_kde
from scipy import interpolate

from clustering.utils import *


def genrand(data,n,cosmo,width=.2,plot=True,plot_filename=None):
	'''
	generates random catalog with random sky distribution and redshift
	To filter based on the BASS sensitivity map, set 'use_BASS_sens_map' to True
	
	'''
	goodz=data['z']>0
	if 'weight' in data.dtype.names:
		weights = data['weight'][goodz]
	else: weights=None
	d = data[goodz]
	z_arr = d['z']
	ra_arr = d['ra']
	dec_arr = d['dec']
	ur,uind = np.unique(d['ra'],return_index=True)
	udata = d[uind]
	ndata = len(udata)
	
	#generate random redshifts
	n_rand = int(round(n*ndata))
	z_grid = np.linspace(min(z_arr), max(z_arr), 1000)
	kde = weighted_gaussian_kde(z_arr, bw_method=width, weights=weights)
	kdepdfz=kde.evaluate(z_grid)
	zr_arr = generate_rand_from_pdf(pdf=kdepdfz, num=n_rand, x_grid=z_grid)
	
	#generate sky coords
	ind = np.random.randint(ndata, size=n_rand)
	rar_arr = udata['ra'][ind]
	decr_arr = udata['dec'][ind]
	
	temp = list(zip(zr_arr,rar_arr,decr_arr))
	rcat = np.zeros((len(zr_arr),), dtype=[('z', '<f8'),('ra', '<f8'),('dec', '<f8')])
	rcat[:] = temp

	randoms = rcat
	rcdists = np.array([cosmo.comoving_distance(z).value for z in randoms['z']])*cosmo.h
	randoms = append_fields(randoms, 'cdist', rcdists)
	random = np.array(randoms)
	
	print('number of randoms:', len(randoms))

	if plot:
		plot_zdist(d,randoms,z_grid,kdepdfz,plot_filename,weights=weights)

	return randoms


def generate_rand_from_pdf(pdf, num, x_grid):
	cdf = np.cumsum(pdf)
	cdf = cdf / cdf[-1]
	values = np.random.rand(num)
	value_bins = np.searchsorted(cdf, values)
	random_from_cdf = x_grid[value_bins]
	return random_from_cdf


def plot_zdist(data,randoms,z_grid,kdepdf,filename=None,weights=None):
	nd = len(np.unique(data['ra']))
	nr = len(randoms)
	#if weights is None:
	#	plt.hist(data['z'], histtype='stepfilled', bins=20,alpha=0.2,color='k',normed=True,label='data \n N='+str(nd))
	#else:
	plt.hist(data['z'], histtype='stepfilled', bins=20,alpha=0.2,color='k',normed=True,label='data \n N='+str(nd),weights=weights)
	plt.hist(randoms['z'], histtype='step', bins=20,alpha=0.5,normed=True,label='randoms\n N='+str(nr))
	plt.plot(z_grid, kdepdf, color='g', alpha=0.5, lw=3,label='smoothed PDF')
	plt.xlabel('z')
	plt.legend(loc='best', frameon=False)
	plt.tight_layout()
	if filename:
		plt.savefig(filename)