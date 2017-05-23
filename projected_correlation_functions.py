from __future__ import division

import numpy as np 
from numpy.lib.recfunctions import append_fields
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
from astropy import units as u
import sys

from clustering.utils import *
from clustering.error import *
from Corrfunc.mocks import DDrppi_mocks

__author__ = "Meredith Powell <meredith.powell@yale.edu>"


def auto_wp(data, randoms, bins, pimax, m, estimator='L'):

	rpwidths=[]
	nbins = len(bins)-1
	for i in np.arange(nbins):
		rpwidths.append(bins[i+1] - bins[i])
	rp = bins[1:] - 0.5*np.array(rpwidths)

	wp = wp_dd(data=data, randoms=randoms, bins=bins, pimax=pimax, estimator=estimator)
	wp_err,cov=auto_jackknife(d=data,r=randoms,m=m,pimax=pimax,bins=bins,estimator=estimator)

	return rp,wp,wp_err,cov


def cross_wp(d1, d2, r2, bins, pimax, m, r1=None, estimator='L'):

	rpwidths=[]
	nbins = len(bins)-1
	for i in np.arange(nbins):
		rpwidths.append(bins[i+1] - bins[i])
	rp = bins[1:] - 0.5*np.array(rpwidths)

	wp = wp_d1d2(d1=d1, d2=d2, r2=r2, bins=bins, pimax=pimax, r1=r1, estimator=estimator)
	wp_err,cov=cross_jackknife(d1=d1,d2=d2,r1=r1,r2=r2,m=m,pimax=pimax,bins=bins,estimator=estimator)

	return rp,wp,wp_err,cov







