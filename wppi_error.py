from __future__ import division

import numpy as np 
from numpy.lib.recfunctions import append_fields
from astropy.table import Table, vstack
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
from astropy import units as u
import sys
from joblib import Parallel, delayed

from AGNclustering.wppi_utils import *
from AGNclustering.error import _BASS_block, _AO13_block, _S82X_block, _XXL_block, _Comb_block
from Corrfunc.mocks import DDrppi_mocks

def wppi_auto_jackknife(d,r,m,pimax,bins,pibins,estimator,covariance=True,survey=None):
	#print('beginning jackknifes')
	if survey=='BASS':
		dblocks=np.array(_BASS_block(d,m),dtype=object)
		rblocks=np.array(_BASS_block(r,m),dtype=object)
	elif survey=='S82X':
		dblocks=np.array(_S82X_block(d,m),dtype=object)
		rblocks=np.array(_S82X_block(r,m),dtype=object)
	elif survey=='AO13':
		dblocks=np.array(_AO13_block(d,m),dtype=object)
		rblocks=np.array(_AO13_block(r,m),dtype=object)
	elif survey=='XXL':
		dblocks=np.array(_XXL_block(d,m),dtype=object)
		rblocks=np.array(_XXL_block(r,m),dtype=object)
	elif survey=='Comb':
		dblocks=np.array(_Comb_block(d,m),dtype=object)
		rblocks=np.array(_Comb_block(r,m),dtype=object)
	else:
		print('No valid survey for error estimation')
		return
	M=len(dblocks)
	print('Using ',M,' jacknife samples')
# 	nbins = len(bins)-1
	npibins = len(pibins)
	#NON-PARALLELIZED JACKKNIFE:
	wp_arr = np.empty(npibins)
	for i in np.arange(M):
		print(str(i+1),'/',str(M))
		new_dblocks = np.delete(dblocks,i,0)
		new_rblocks = np.delete(rblocks,i,0)
		dset = np.concatenate(new_dblocks)
		rset = np.concatenate(new_rblocks)
		wp_t = wppi_dd(data=dset,randoms=rset,pimax=pimax,bins=bins,pibins=pibins,estimator=estimator)
		wp_arr=np.vstack((wp_arr,wp_t))
	wp_arr=wp_arr[1:]
	
	#PARALLELIZED JACKKNIFE COMPUTATION:
	#wp_arr = np.array(Parallel(n_jobs=-1)(delayed(PAjackknife)(i,dblocks,rblocks,pimax=pimax,bins=bins,estimator=estimator) for i in np.arange(M)))

	err =np.sqrt(((M-1.)/M) * np.sum( (wp_arr - np.average(wp_arr,axis=0))**2, axis=0))
	if covariance:
		cov = np.zeros(shape = (npibins,npibins))
		for i in np.arange(npibins):
			for j in np.arange(npibins):
				cov[i,j] = ((M-1.)/M)* np.sum( (wp_arr[:,i] - np.average(wp_arr,axis=0)[i]) * (wp_arr[:,j] - np.average(wp_arr,axis=0)[j]), axis=0)
		return err,cov
	else: return err

    
def wppi_cross_jackknife(d1,d2,r1,r2,m,pimax,bins,pibins,estimator,covariance=True,survey=None,weights1=None,weights2=None):
	#print('beginning jackknifes')
	if survey=='BASS':
		d1blocks=np.array(_BASS_block(d1,m),dtype=object)
		d2blocks=np.array(_BASS_block(d2,m),dtype=object)
		r1blocks=np.array(_BASS_block(r1,m),dtype=object)
		r2blocks=np.array(_BASS_block(r2,m),dtype=object)
	elif survey=='S82X':
		d1blocks=np.array(_S82X_block(d1,m),dtype=object)
		d2blocks=np.array(_S82X_block(d2,m),dtype=object)
		r1blocks=np.array(_S82X_block(r1,m),dtype=object)
		r2blocks=np.array(_S82X_block(r2,m),dtype=object)
	elif survey=='AO13':
		d1blocks=np.array(_AO13_block(d1,m),dtype=object)
		d2blocks=np.array(_AO13_block(d2,m),dtype=object)
		r1blocks=np.array(_AO13_block(r1,m),dtype=object)
		r2blocks=np.array(_AO13_block(r2,m),dtype=object)
	elif survey=='Comb':
		d1blocks=np.array(_Comb_block(d1,m),dtype=object)
		d2blocks=np.array(_Comb_block(d2,m),dtype=object)
		r1blocks=np.array(_Comb_block(r1,m),dtype=object)
		r2blocks=np.array(_Comb_block(r2,m),dtype=object)
	else:
		print('No valid survey for error estimation')
		return
	M=len(d1blocks)
	print('Using ',M,' jacknife samples')
	npibins = len(pibins)

	#NON-PARALLELIZED JACKKNIFE:
	wp_arr = np.empty(npibins)
	#compute wp without one block
	for i in np.arange(M):
		print(str(i+1),'/',str(M))
		new_d1blocks = np.delete(d1blocks,i,0)
		new_d2blocks = np.delete(d2blocks,i,0)
		if r1 is not None:
			new_r1blocks = np.delete(r1blocks,i,0)
		new_r2blocks = np.delete(r2blocks,i,0)
		d1set = np.concatenate(new_d1blocks)
		d2set = np.concatenate(new_d2blocks)
		if r1 is not None:
			r1set = np.concatenate(new_r1blocks)
		else:
			r1set=None
		r2set = np.concatenate(new_r2blocks)
		wp_t = wppi_d1d2(d1=d1set, d2=d2set, r1=r1set, r2=r2set, bins=bins, pimax=pimax, pibins=pibins, estimator=estimator)
		wp_arr=np.vstack((wp_arr,wp_t))
	wp_arr=wp_arr[1:]

	#PARALLELIZED JACKKNIFE COMPUTATION:
	#wp_arr = np.array(Parallel(n_jobs=-1)(delayed(PCjackknife)(i,d1blocks,d2blocks,r2blocks,pimax=pimax,bins=bins,estimator=estimator,r1blocks=r1blocks) for i in np.arange(M)))

	err =np.sqrt(((M-1.)/M) * np.sum( (wp_arr - np.average(wp_arr,axis=0))**2, axis=0))
	if covariance:
		cov = np.zeros(shape = (npibins,npibins))
		for i in np.arange(npibins):
			for j in np.arange(npibins):
				cov[i,j] = ((M-1.)/M)* np.sum( (wp_arr[:,i] - np.average(wp_arr,axis=0)[i]) * (wp_arr[:,j] - np.average(wp_arr,axis=0)[j]), axis=0)
		return err,cov
	else: return err
