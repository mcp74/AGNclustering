from __future__ import division

import numpy as np 
from numpy.lib.recfunctions import append_fields
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
from astropy import units as u
import sys

from clustering.utils import *
from Corrfunc.mocks import DDrppi_mocks

def auto_jackknife(d,r,m,pimax,bins,estimator,covariance=True):
	print 'beginning jackknifes'
	dblocks=_block(d,m)
	rblocks=_block(r,m)
	M=len(dblocks)
	nbins = len(bins)-1
	wp_arr = np.empty(nbins)
	
	#compute wp without one block
	for i in np.arange(M):
		print str(i+1),'/',str(M)
		new_dblocks = np.delete(dblocks,i,0)
		new_rblocks = np.delete(rblocks,i,0)
		dset = np.concatenate(new_dblocks)
		rset = np.concatenate(new_rblocks)
		wp_t = wp_dd(data=dset,randoms=rset,pimax=pimax,bins=bins,estimator=estimator)
		wp_arr=np.vstack((wp_arr,wp_t))
	wp_arr=wp_arr[1:]

	err =np.sqrt(((M-1.)/M) * np.sum( (wp_arr - np.average(wp_arr,axis=0))**2, axis=0))
	if covariance:
		cov = np.zeros(shape = (nbins,nbins))
		for i in np.arange(nbins):
			for j in np.arange(nbins):
				cov[i,j] = ((M-1.)/M)* np.sum( (wp_arr[:,i] - np.average(wp_arr,axis=0)[i]) * (wp_arr[:,j] - np.average(wp_arr,axis=0)[j]), axis=0)
		return err,cov
	else: return err

def cross_jackknife(d1,d2,r1,r2,m,pimax,bins,estimator,covariance=True):
	print 'beginning jackknifes'
	d1blocks=_block(d1,m)
	d2blocks=_block(d2,m)
	r1blocks=_block(r1,m)
	r2blocks=_block(r2,m)

	M=len(d1blocks)
	nbins = len(bins)-1
	wp_arr = np.empty(nbins)
	
	#compute wp without one block
	for i in np.arange(M):
		print str(i+1),'/',str(M)
		new_d1blocks = np.delete(d1blocks,i,0)
		new_d2blocks = np.delete(d2blocks,i,0)
		new_r1blocks = np.delete(r1blocks,i,0)
		new_r2blocks = np.delete(r2blocks,i,0)
		d1set = np.concatenate(new_d1blocks)
		d2set = np.concatenate(new_d2blocks)
		r1set = np.concatenate(new_r1blocks)
		r2set = np.concatenate(new_r2blocks)
		wp_t = wp_d1d2(d1=d1set, d2=d2set, r1=r1set, r2=r2set, bins=bins, pimax=pimax, estimator=estimator)
		wp_arr=np.vstack((wp_arr,wp_t))
	wp_arr=wp_arr[1:]

	err =np.sqrt(((M-1.)/M) * np.sum( (wp_arr - np.average(wp_arr,axis=0))**2, axis=0))
	if covariance:
		cov = np.zeros(shape = (nbins,nbins))
		for i in np.arange(nbins):
			for j in np.arange(nbins):
				cov[i,j] = ((M-1.)/M)* np.sum( (wp_arr[:,i] - np.average(wp_arr,axis=0)[i]) * (wp_arr[:,j] - np.average(wp_arr,axis=0)[j]), axis=0)
		return err,cov
	else: return err



def _block(cat,m):

	blocks=[]
	lens=[]

	ra0=0
	ra_int=(360./m)
	dec_int=(180./m)
	#dec_int=(2./m)
	for i in range(m):
		dec0=-90
		#dec0=-1
		for j in range(m):
			ra_min=ra0
			ra_max=ra0+ra_int
			dec_min=dec0
			dec_max=dec0+dec_int
			#dec_min = (np.arcsin(dec0)*u.radian).to(u.degree).value
			#dec_max = (np.arcsin(dec0+dec_int)*u.radian).to(u.degree).value
			block=cat[(cat['ra']<ra_max)&(cat['ra']>=ra_min)&\
					(cat['dec']<dec_max)&(cat['dec']>=dec_min)]
			dec0=dec_max
			blocks.append(block)
			lens.append(len(block))
			#print ra_min,dec_min
		ra0=ra_max
	return blocks
