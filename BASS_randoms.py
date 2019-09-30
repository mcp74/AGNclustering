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
from os.path import expanduser

from clustering.kde import weighted_gaussian_kde



def genrand(data,n,cosmo,width=.2,scoords='galactic',use_BASS_sens_map=False,data_path='/Users/meredithpowell/Dropbox/Data/BASS/sensitivity_maps/',plot=True,plot_filename=None):
	'''
	generates random catalog with random sky distribution and redshift
	To filter based on the BASS sensitivity map, set 'use_BASS_sens_map' to True
	
	'''
	z_arr = data['z']
	l_arr = data['l']
	b_arr = data['b']
	
	#generate random redshifts
	if use_BASS_sens_map is True:
		n_rand = int(round(n*len(data)*1.2))
	else: n_rand = int(round(n*len(data)))
	z_grid = np.linspace(min(z_arr), max(z_arr), 1000)
	kde = weighted_gaussian_kde(z_arr, bw_method=width, weights=None)
	kdepdfz=kde.evaluate(z_grid)
	zr_arr = generate_rand_from_pdf(pdf=kdepdfz, num=n_rand, x_grid=z_grid)
	
	#generate random sky coord
	rar_arr = (np.random.rand(len(zr_arr))*360)
	decrad = np.arcsin(np.random.rand(len(zr_arr))*2.-1) * u.radian
	decr_arr = decrad.to(u.degree).value
	
	if scoords=='galactic':
		ran_coords = SkyCoord(ra = rar_arr*u.degree, dec = decr_arr*u.degree)
		ran_galcoords = ran_coords.galactic
		lr_arr = ran_galcoords.l.value
		br_arr = ran_galcoords.b.value
	
	temp = list(zip(zr_arr,rar_arr,decr_arr,lr_arr,br_arr))
	rcat = np.zeros((len(zr_arr),), dtype=[('z', '<f8'),('ra', '<f8'),('dec', '<f8'),('l', '<f8'),\
									   ('b', '<f8')])
	rcat[:] = temp
	
	if use_BASS_sens_map is True:
		if 'flux' not in data.dtype.names:
			print('no flux data in catalog found to filter based on sensitivity')
		else: 
			rcat = BASS_sensitivity_filter(data_path,data,rcat)

	randoms=rcat
	rcdists = np.array([cosmo.comoving_distance(z).value for z in randoms['z']])*cosmo.h
	randoms = append_fields(randoms, 'cdist', rcdists)
	randoms=np.array(randoms)
	
	print('number of randoms:', len(randoms))

	if plot:
		plot_zdist(data,randoms,z_grid,kdepdfz,plot_filename)

	return randoms


def BASS_sensitivity_filter(path,data,rcat):

	flux_arr = data['flux']
	n_rand = len(rcat)

	#generate random fluxes
	log_flux_grid = np.linspace(min(np.log10(flux_arr)), max(np.log10(flux_arr)), 1000)
	kde = weighted_gaussian_kde(np.log10(flux_arr), weights=None)
	kdepdff=kde.evaluate(log_flux_grid)
	log_fluxr_arr = generate_rand_from_pdf(pdf=kdepdff, num=n_rand, x_grid=log_flux_grid)
	fluxr_arr = 10**(log_fluxr_arr)

	rcat = append_fields(rcat, 'flux', fluxr_arr)

	smaps,wcses=get_BASSsmap(path)
	
	#filter based on sensitivity
	good=[]
	for i,r in enumerate(rcat):
		l=r['l']
		b=r['b']
		flux = r['flux'] #ergs/s/cm^2
		px,py,sind=BASSmap_ind(l,b,wcses)
		sens_map = smaps[sind]
		try:
			sensitivity = sens_map[px,py]*2.39e-8*4.8 #in ergs/s/cm^-2
		except IndexError:
			print(l,b)
		if flux>sensitivity:
			good = np.append(good,i)
	randoms=rcat[good.astype(int)]
	
	return randoms


def get_BASSsmap(direc):
	'''Enter Galactic coordinates'''
	from astropy.wcs import WCS
	w0 = WCS(direc + 'swiftbat_bkgstd_70month4_c0_tot_crab.fits')
	w1 = WCS(direc + 'swiftbat_bkgstd_70month4_c1_tot_crab.fits')
	w2 = WCS(direc + 'swiftbat_bkgstd_70month4_c2_tot_crab.fits')
	w3 = WCS(direc + 'swiftbat_bkgstd_70month4_c3_tot_crab.fits')
	w4 = WCS(direc + 'swiftbat_bkgstd_70month4_c4_tot_crab.fits')
	w5 = WCS(direc + 'swiftbat_bkgstd_70month4_c5_tot_crab.fits')

	h0 = fits.open(direc + 'swiftbat_bkgstd_70month4_c0_tot_crab.fits')
	h1 = fits.open(direc + 'swiftbat_bkgstd_70month4_c1_tot_crab.fits')
	h2 = fits.open(direc + 'swiftbat_bkgstd_70month4_c2_tot_crab.fits')
	h3 = fits.open(direc + 'swiftbat_bkgstd_70month4_c3_tot_crab.fits')
	h4 = fits.open(direc + 'swiftbat_bkgstd_70month4_c4_tot_crab.fits')
	h5 = fits.open(direc + 'swiftbat_bkgstd_70month4_c5_tot_crab.fits')
	s0 = h0[0].data
	s1 = h1[0].data
	s2 = h2[0].data
	s3 = h3[0].data
	s4 = h4[0].data
	s5 = h5[0].data
	smaps = [s0,s1,s2,s3,s4,s5]
	wcses = [w0,w1,w2,w3,w4,w5]
	return smaps,wcses

def BASSmap_ind(l,b,wcses):
	w0=wcses[0]
	w1=wcses[1]
	w2=wcses[2]
	w3=wcses[3]
	w4=wcses[4]
	w5=wcses[5]
	py,px = w0.wcs_world2pix(l,b,1)
	i=0
	if (px<0)|(px>1997)|(py<0)|(py>1997):
		py,px = w1.wcs_world2pix(l,b,1)
		i=1
	if (px<0)|(px>1997)|(py<0)|(py>1997):
		py,px = w2.wcs_world2pix(l,b,1)
		i=2
	if (px<0)|(px>1997)|(py<0)|(py>1997):
		py,px = w3.wcs_world2pix(l,b,1)
		i=3
	if (px<0)|(px>1997)|(py<0)|(py>1997):
		py,px = w4.wcs_world2pix(l,b,1)
		i=4
	if (px<0)|(px>1997)|(py<0)|(py>1997):
		py,px = w5.wcs_world2pix(l,b,1)
		i=5
	px = np.round(px).astype(int)
	py = np.round(py).astype(int)
	return px,py,i


def generate_rand_from_pdf(pdf, num, x_grid):
	cdf = np.cumsum(pdf)
	cdf = cdf / cdf[-1]
	values = np.random.rand(num)
	value_bins = np.searchsorted(cdf, values)
	random_from_cdf = x_grid[value_bins]
	return random_from_cdf


def plot_zdist(data,randoms,z_grid,kdepdf,filename=None):
	nd = len(data)
	nr = len(randoms)
	plt.hist(data['z'], histtype='stepfilled', bins=20,alpha=0.2,color='k',density=True,label='data \n N='+str(nd))
	plt.hist(randoms['z'], histtype='step', bins=20,alpha=0.5,density=True,label='randoms\n N='+str(nr))
	plt.plot(z_grid, kdepdf, color='g', alpha=0.5, lw=3,label='smoothed PDF')
	plt.xlabel('z',fontsize=14)
	plt.legend(loc='best', frameon=False,fontsize=14)
	plt.tight_layout()
	if filename:
		plt.savefig(filename)


