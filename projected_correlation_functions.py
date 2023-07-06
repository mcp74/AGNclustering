from __future__ import division

import numpy as np 
from numpy.lib.recfunctions import append_fields
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
from astropy import units as u
import sys

from AGNclustering.utils import *
from AGNclustering.error import *
from Corrfunc.mocks import DDrppi_mocks

__author__ = "Meredith Powell <meredith.powell@yale.edu>"


def auto_wp(data, randoms, bins, pimax, m, estimator='L',cosmo=None,survey='BASS'):
	'''
	Computes the projected autocorrelation function a catalog of data with an associated random catalog
	Utilizes the pair counter from CorrFunc (https://github.com/manodeep/Corrfunc)

	data: structured array of data with columns 'ra', 'dec', and either 'z' (redshift) or 'cdist' (comoving distance). For weights, have a 'weight' column
	randoms: structured array of randoms with columns 'ra', 'dec', and either 'z' (redshift) or 'cdist' (comoving distance)
	bins: array of boundaries defining the bins of scale (perpendicular to the line of sight), in units of Mpc/h
	pimax: maximum distance along the line of sight defining the projection integral length-scale, in units of Mpc/h
	m: number of jacknife samples for error estimation
	estimator: either 'L' or 'Landy' for the Landy-Szalay estimator, or 'P' or 'Peebles' for Peebles estimator
	cosmo: astropy cosmology object, used if no 'cdist' column in either data or random array. if cosmo=None, and there is no 'cdist' column in data arrays, flat LCDM cosmology is used.
	survey: 'BASS' or 'S82X', for error estimation
	'''
	data = np.array(data)
	randoms = np.array(randoms)

	rpwidths=[]
	nbins = len(bins)-1
	for i in np.arange(nbins):
		rpwidths.append(bins[i+1] - bins[i])
	rp = bins[1:] - 0.5*np.array(rpwidths)

	if 'cdist' not in data.dtype.names:
		data = z_to_cdist(data,cosmo)
	if 'cdist' not in randoms.dtype.names:
		randoms = z_to_cdist(randoms,cosmo)
	if 'weight' in data.dtype.names:
		print('Found weights. Will output weighted correlation function')

	wp = wp_dd(data=data, randoms=randoms, bins=bins, pimax=pimax, estimator=estimator)
	wp_err,cov=auto_jackknife(d=data,r=randoms,m=m,pimax=pimax,bins=bins,estimator=estimator,survey=survey)

	return rp,wp,wp_err,cov


def cross_wp(d1, d2, r2, bins, pimax, m, r1=None, estimator='L',cosmo=None,survey='BASS'):
	'''
	Computes the projected crosscorrelation function between two catalogs of data with associated random catalogs.
	Utilizes the pair counter from CorrFunc (https://github.com/manodeep/Corrfunc)

	d1: structured array of first data catalog with columns 'ra', 'dec', and either 'z' (redshift) or 'cdist' (comoving distance). For weights, have a 'weight' column
	r1: structured array of d1 randoms with columns 'ra', 'dec', and either 'z' (redshift) or 'cdist' (comoving distance). Required for Landy-Szalay estimator.
	d2: structured array of second data catalog with columns 'ra', 'dec', and either 'z' (redshift) or 'cdist' (comoving distance). For weights, have a 'weight' column
	r2: structured array of d2 randoms with columns 'ra', 'dec', and either 'z' (redshift) or 'cdist' (comoving distance)
	bins: array of boundaries defining the bins of scale (perpendicular to the line of sight), in units of Mpc/h
	pimax: maximum distance along the line of sight defining the projection integral length-scale, in units of Mpc/h
	m: number of jacknife samples for error estimation
	estimator: either 'L' or 'Landy' for the Landy-Szalay estimator, or 'P' or 'Peebles' for Peebles estimator. If Landy-Szalay, random catalog for d1 is needed.
	cosmo: astropy cosmology object, used if no 'cdist' column in catalog arrays. if cosmo=None, and there is no 'cdist' column in data arrays, flat LCDM cosmology is used.
	survey: 'BASS' or 'S82X', for error estimation
	'''

	d1 = np.array(d1)
	d2 = np.array(d2)
	if (r1 is not None):
		r1 = np.array(r1)
	r2 = np.array(r2)

	rpwidths=[]
	nbins = len(bins)-1
	for i in np.arange(nbins):
		rpwidths.append(bins[i+1] - bins[i])
	rp = bins[1:] - 0.5*np.array(rpwidths)

	if 'cdist' not in d1.dtype.names:
		d1 = z_to_cdist(d1,cosmo)
	if 'cdist' not in d2.dtype.names:
		d2 = z_to_cdist(d2,cosmo)
	if (r1 is not None):
		if ('cdist' not in r1.dtype.names):
			r1 = z_to_cdist(r1,cosmo)
	if 'cdist' not in r2.dtype.names:
		r2 = z_to_cdist(r2,cosmo)
	if ('weight' in d1.dtype.names) or ('weight' in d2.dtype.names):
		print('Found weights. Will output weighted correlation function')

	wp,zp = wp_d1d2(d1=d1, d2=d2, r2=r2, bins=bins, pimax=pimax, r1=r1, estimator=estimator)
	wp_err,cov=cross_jackknife(d1=d1,d2=d2,r1=r1,r2=r2,m=m,pimax=pimax,bins=bins,estimator=estimator,survey=survey)

	return rp,wp,wp_err,cov,zp







