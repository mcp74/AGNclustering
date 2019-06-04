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
from astropy.wcs import WCS

from clustering.utils import *
from clustering.XMM_XXL_utils import *

direc = '/Users/meredithpowell/Dropbox/clustering/XXL/'

def genrand(data,n,cosmo,width=.2,use_S82X_sens_map=True,plot=True,plot_filename=None):
	'''
	generates random catalog with random sky distribution and redshift
	To filter based on the BASS sensitivity map, set 'use_BASS_sens_map' to True
	
	'''
	goodz=data['z']>0
	if 'weight' in data.dtype.names:
		weights = data['weight'][goodz]
	else: weights=None
	z_arr = data['z'][goodz]
	ra_arr = data['ra'][goodz]
	dec_arr = data['dec'][goodz]
	ndata = len(np.unique(data['ra'][goodz]))
	
	#generate random redshifts
	if use_S82X_sens_map is True:
		n_rand = int(round(n*ndata*3.2))
	else: n_rand = int(round(n*ndata))
	z_grid = np.linspace(min(z_arr), max(z_arr), 1000)
	kde = weighted_gaussian_kde(z_arr, bw_method=width, weights=weights)
	kdepdfz=kde.evaluate(z_grid)
	zr_arr = generate_rand_from_pdf(pdf=kdepdfz, num=n_rand, x_grid=z_grid)
	
	ra_max = 39.1
	ra_min = 29.8
	dec_min = -7.5
	dec_max = -2.5
	rar_arr = (np.random.rand(len(zr_arr))*(ra_max - ra_min))+ra_min
	decr_arr = (np.random.rand(len(zr_arr))*(dec_max - dec_min))+dec_min    
	
	temp = list(zip(zr_arr,rar_arr,decr_arr))
	rcat = np.zeros((len(zr_arr),), dtype=[('z', '<f8'),('ra', '<f8'),('dec', '<f8')])
	rcat[:] = temp
	
	rcat = XXL_sensitivity_filter(rcat)

	randoms=rcat[rcat['flux']>1e-14]
	rcdists = np.array([cosmo.comoving_distance(z).value for z in randoms['z']])*cosmo.h
	randoms = append_fields(randoms, 'cdist', rcdists)
	randoms=np.array(randoms)
	
	print('number of randoms:', len(randoms))

	if plot:
		plot_zdist(data[goodz],randoms,z_grid,kdepdfz,plot_filename,weights=weights)

	return randoms

def XXL_sensitivity_filter(rcat):

	n_rand = len(rcat)

	#get logN-logS distribution:
	#fn = get_lognlogs()

	#assign flux from data distribution
	t = Table.read(direc+'catalogs/xxl_xz-matched.fits')
	xxldat = np.array(t)
	f_arr = np.log10(xxldat['flux_full'][xxldat['flux_full']>0])
	f_grid = np.linspace(min(f_arr), max(f_arr), 1000)
	kde = weighted_gaussian_kde(f_arr, bw_method=0.1)
	kdepdff=kde.evaluate(f_grid)
	log_rflux_arr = generate_rand_from_pdf(pdf=kdepdff, num=n_rand, x_grid=f_grid)
	fluxr_arr = 10**(log_rflux_arr)

	#rcat = append_fields(rcat, 'flux', fluxr_arr)
	
	#filter based on sensitivity
	smap,wcss = get_XXLsmap()
	bmap,wcsb = get_XXLbmap()
	emap,wcse = get_XXLemap()
	good=[]
	for i,r in enumerate(rcat):
		ra=r['ra']
		dec=r['dec']
		#flux = r['flux'] #ergs/s/cm^2
		flux = fluxr_arr[i]

		pys,pxs = wcss.wcs_world2pix(ra,dec,1)
		pxs = np.round(pxs).astype(int)
		pys = np.round(pys).astype(int)

		pyb,pxb = wcsb.wcs_world2pix(ra,dec,1)
		pxb = np.round(pxb).astype(int)
		pyb = np.round(pyb).astype(int)

		pye,pxe = wcse.wcs_world2pix(ra,dec,1)
		pxe = np.round(pxe).astype(int)
		pye = np.round(pye).astype(int)

		counts = flux * 0.7 * emap[pxe,pye]*1e11 + bmap[pxb,pyb]
		sensitivity = smap[pxs,pys]
		if (counts>sensitivity) & (sensitivity>0) & (bmap[pxb,pyb]!=0) & boss_footprint(ra=ra,dec=dec):
			good = np.append(good,i)
	randoms=rcat[good.astype(int)]
	f_arr = fluxr_arr[good.astype(int)]
	randoms = append_fields(randoms, 'flux', f_arr)
	
	return randoms


def get_XXLsmap():
	w0 = WCS(direc + 'sensitivity_maps/full_sense.fits')
	h0 = fits.open(direc + 'sensitivity_maps/full_sense.fits')
	s0 = h0[0].data
	smap = s0
	wcs = w0
	return smap,wcs

def get_XXLbmap():
	w0 = WCS(direc + 'sensitivity_maps/full_bkgeef.fits')
	h0 = fits.open(direc + 'sensitivity_maps/full_bkgeef.fits')
	s0 = h0[0].data
	bmap = s0
	wcs = w0
	return bmap,wcs

def get_XXLemap():
	w0 = WCS(direc + 'sensitivity_maps/full_expeef.fits')
	h0 = fits.open(direc + 'sensitivity_maps/full_expeef.fits')
	s0 = h0[0].data
	emap = s0
	wcs = w0
	return emap,wcs

def generate_rand_from_pdf(pdf, num, x_grid):
	cdf = np.cumsum(pdf)
	cdf = cdf / cdf[-1]
	values = np.random.rand(num)
	value_bins = np.searchsorted(cdf, values)
	random_from_cdf = x_grid[value_bins]
	return random_from_cdf

def plot_zdist(data,randoms,z_grid,kdepdf,filename=None,weights=None):
	nd = len(data)
	nr = len(randoms)
	#if weights is None:
	#   plt.hist(data['z'], histtype='stepfilled', bins=20,alpha=0.2,color='k',normed=True,label='data \n N='+str(nd))
	#else:
	plt.hist(data['z'], histtype='stepfilled', bins=20,alpha=0.2,color='k',normed=True,label='data \n N='+str(nd),weights=weights)
	plt.hist(randoms['z'], histtype='step', bins=20,alpha=0.5,normed=True,label='randoms\n N='+str(nr))
	plt.plot(z_grid, kdepdf, color='g', alpha=0.5, lw=3,label='smoothed PDF')
	plt.xlabel('z')
	plt.legend(loc='best', frameon=False)
	plt.tight_layout()
	if filename:
		plt.savefig(filename)