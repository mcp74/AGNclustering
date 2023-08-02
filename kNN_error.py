from __future__ import division

import numpy as np 
from numpy.lib.recfunctions import append_fields
from astropy.table import Table, vstack
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
from astropy import units as u
import sys
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from AGNclustering.wppi_utils import *
from AGNclustering.error import _BASS_block, _AO13_block, _S82X_block, _XXL_block, _Comb_block
from AGNclustering.KNN_stuff import CDFkNN_rp_pi

def kNN_jacknife(d,r,rs,pis,kneighbors,m=5,rpbool=True,covariance=True,survey='BASS'):
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
	nrs = len(rs)
	npis = len(pis)
	errors=np.empty(nrs)
	covar=np.zeros(shape = (nrs,nrs))
	#NON-PARALLELIZED JACKKNIFE:
	cdf_arr = np.empty(nrs)
	for i in np.arange(M):
		print(str(i+1),'/',str(M))
		new_dblocks = np.delete(dblocks,i,0)
		new_rblocks = np.delete(rblocks,i,0)
		dset = np.concatenate(new_dblocks)
		rset = np.concatenate(new_rblocks)
		cdf_ts = CDFkNN_rp_pi(rs=rs,pis=pis,xgalfull=dset,xrandfull=rset,kneighbors=kneighbors)
		for j in range(kneighbors):
			n=cdf_ts[j]
#             For kNN(rp)
			if rpbool == True:
				cdf_t= n[-1]
			else:
#             For kNN(pi)
				cdf_t=[row[-1] for row in n]
    
			cdf_arr=np.vstack((cdf_arr,cdf_t))
	cdf_arr=cdf_arr[1:]

	for i in range(kneighbors):
		for k in range(len(cdf_arr)):
			cdf_a=cdf_arr[i::kneighbors]
		errs = np.sqrt(((M-1.)/M) * np.sum( (cdf_a - np.average(cdf_a,axis=0))**2, axis=0))
		errors = np.vstack((errors,errs))
   
	errors=errors[1:]
                
	#PARALLELIZED JACKKNIFE COMPUTATION:
	#wp_arr = np.array(Parallel(n_jobs=-1)(delayed(PAjackknife)(i,dblocks,rblocks,pimax=pimax,bins=bins,estimator=estimator) for i in np.arange(M)))


	if covariance:
# 		for a in range(kneighbors):
# 			for b in range(len(cdf_arr)):
# 				cdf_a=cdf_arr[a::kneighbors]
		cov = np.zeros(shape = (nrs,nrs))
		for i in np.arange(nrs):
			for j in np.arange(nrs):
				cov[i,j] = ((M-1.)/M)* np.sum( (cdf_a[:,i] - np.average(cdf_a,axis=0)[i]) * (cdf_a[:,j] - np.average(cdf_a,axis=0)[j]), axis=0)
		return errors,cov
	else: return errors
    
def func(a):
	return (a**2)

def func2(a,b):
	if b !=0:
		return (a/b)
	else:
		return 0.0

def func3(a):
	return math.sqrt(a)

def kNN_ratio_error(d1,d2,d1err,d2err):
	vfunc = np.vectorize(func)
	afunc = np.vectorize(func2)
	bfunc = np.vectorize(func3)
	if len(d1)!=len(d2):
		sys.exit("Error arrays different lengths")
	ratio_error = np.zeros((len(d1),len(d1[0])))
	for i in range(len(d1)):                   
            ratio_error[i]=bfunc(vfunc(afunc(d1err[i],d1[i]))+vfunc(afunc(d2err[i],d2[i])))
	return ratio_error