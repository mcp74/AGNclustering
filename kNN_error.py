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
from AGNclustering.angular_kNN import CDFkNN_theta

def angular_kNN_jacknife(d, r, angles, kneighbors, angleslist, m=5, concatenate = False, covariance=True, survey='BASS'):
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
	nrs = len(angles)
	errors=np.empty(nrs)
	covar=np.zeros(shape = (nrs,nrs))
	#NON-PARALLELIZED JACKKNIFE:
	cdf_arr = np.empty(nrs)
    
	if(concatenate == True):
		nrs = 6
		npis = 6
		errors = np.empty(nrs)
		covar = np.zeros(shape = (nrs,nrs))
		cdf_arr = np.empty(nrs)
        
	for i in np.arange(M):
		print(str(i+1),'/',str(M))
		new_dblocks = np.delete(dblocks,i,0)
		new_rblocks = np.delete(rblocks,i,0)
		dset = np.concatenate(new_dblocks)
		rset = np.concatenate(new_rblocks)
        
		decg = dset['dec'] * ((2*np.pi)/360.0)
		rag = dset['ra'] * ((2*np.pi)/360.0)

		decr = rset['dec'] * ((2*np.pi)/360.0)
		rar = rset['ra'] * ((2*np.pi)/360.0)

		agn_angles = np.vstack((decg, rag)).T
		gal_angles = np.vstack((decr, rar)).T
        
		cdf_t = CDFkNN_theta(angles, gal_angles, agn_angles, kneighbors)
		for j in range(kneighbors):
			n=cdf_t[j]  
# 			if (concatenate == True):
# 				n = n[angleslist[j]]
			cdf_arr=np.vstack((cdf_arr,n))
	cdf_arr=cdf_arr[1:]

	for i in range(kneighbors):
		cdf_a=cdf_arr[i::kneighbors]
		errs = np.sqrt(((M-1.)/M) * np.sum( (cdf_a - np.average(cdf_a,axis=0))**2, axis=0))
# 		errs = (np.std(cdf_a,axis=0))
		errors = np.vstack((errors,errs))
   
	errors=errors[1:]
                
	#PARALLELIZED JACKKNIFE COMPUTATION:
	#wp_arr = np.array(Parallel(n_jobs=-1)(delayed(PAjackknife)(i,dblocks,rblocks,pimax=pimax,bins=bins,estimator=estimator) for i in np.arange(M)))

	c=np.empty((nrs-2,nrs-2))
	if concatenate == False:
		if covariance:
			for a in range(kneighbors):
				cdf_a=cdf_arr[a::kneighbors]
				cdf_a=np.delete(cdf_a,-1,axis=1)
				cdf_a=np.delete(cdf_a,0,axis=1)
				cov = np.zeros(shape = (nrs-2,nrs-2))
				for i in np.arange(nrs-2):
					for j in np.arange(nrs-2):
						cov[i,j] = ((M-1.)/M)* np.sum( (cdf_a[:,i] - np.average(cdf_a,axis=0)[i]) * (cdf_a[:,j] - np.average(cdf_a,axis=0)[j]), axis=0)
				if a == 0:
					covar = np.vstack((c[None],cov[None]))
				else:
					covar = np.vstack((covar,cov[None]))
			covar=covar[1:]
			return errors,covar
		else: return errors

	else:
		for a in range(kneighbors):
			cdf_temp=cdf_arr[a::kneighbors]
			cdf_temp=np.delete(cdf_temp,-1,axis=1)
			cdf_temp=np.delete(cdf_temp,0,axis=1)
			if a == 0:
				cdf_a = cdf_temp
			else:
				cdf_a = np.concatenate((cdf_a,cdf_temp),axis=1)
		cov = np.zeros(shape = ((nrs-2)*kneighbors,(nrs-2)*kneighbors))
		for i in np.arange((nrs-2)*kneighbors):
			for j in np.arange((nrs-2)*kneighbors):
				cov[i,j] = ((M-1.)/M)* np.sum( (cdf_a[:,i] - np.average(cdf_a,axis=0)[i]) * (cdf_a[:,j] - np.average(cdf_a,axis=0)[j]), axis=0)
            
		return errors,cov

def block_test(d,r,m,survey='BASS'):
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
	ls=[]
	print('Using ',M,' jacknife samples')
	for i in np.arange(M):
		print(str(i+1),'/',str(M))
		new_dblocks = np.delete(dblocks,i,0)
		new_rblocks = np.delete(rblocks,i,0)
		dset = np.concatenate(new_dblocks)
		rset = np.concatenate(new_rblocks)
		print(len(dset), 'AGN in block')
		ls = np.append(ls,len(dset))
		print(len(rset), 'galaxies in block')
	return ls

def kNN_jacknife(d,r,rs,pis,kneighbors,rslist,rlistbool = False, m=5,rpbool=True,covariance=True,concatenate=False,survey='BASS'):
    
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
    
#     Concatenating Neighbors stuff
	if(rlistbool == True):
		nrs = 4
		npis = 4
		errors = np.empty(nrs)
		covar = np.zeros(shape = (nrs,nrs))
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
				if(rlistbool == True):
					cdf_t = cdf_t[rslist[j]]
			else:
#             For kNN(pi)
				cdf_t=[row[-1] for row in n]
    
			cdf_arr=np.vstack((cdf_arr,cdf_t))
	cdf_arr=cdf_arr[1:]

	for i in range(kneighbors):
		cdf_a=cdf_arr[i::kneighbors]
		errs = np.sqrt(((M-1.)/M) * np.sum( (cdf_a - np.average(cdf_a,axis=0))**2, axis=0))
# 		errs = (np.std(cdf_a,axis=0))
		errors = np.vstack((errors,errs))
   
	errors=errors[1:]
                
	#PARALLELIZED JACKKNIFE COMPUTATION:
	#wp_arr = np.array(Parallel(n_jobs=-1)(delayed(PAjackknife)(i,dblocks,rblocks,pimax=pimax,bins=bins,estimator=estimator) for i in np.arange(M)))

	c=np.empty((nrs,nrs))
	if concatenate == False:
		if covariance:
			for a in range(kneighbors):
				cdf_a=cdf_arr[a::kneighbors]
				cov = np.zeros(shape = (nrs,nrs))
				for i in np.arange(nrs):
					for j in np.arange(nrs):
						cov[i,j] = ((M-1.)/M)* np.sum( (cdf_a[:,i] - np.average(cdf_a,axis=0)[i]) * (cdf_a[:,j] - np.average(cdf_a,axis=0)[j]), axis=0)
				if a == 0:
					covar = np.vstack((c[None],cov[None]))
				else:
					covar = np.vstack((covar,cov[None]))
			covar=covar[1:]
			return errors,covar
		else: return errors

	else:
		for a in range(kneighbors):
			cdf_temp=cdf_arr[a::kneighbors]
			cdf_temp=np.delete(cdf_temp,-1,axis=1)
			if a == 0:
				cdf_a = cdf_temp
			else:
				cdf_a = np.concatenate((cdf_a,cdf_temp),axis=1)
		cov = np.zeros(shape = ((nrs-1)*kneighbors,(nrs-1)*kneighbors))
		for i in np.arange((nrs-1)*kneighbors):
			for j in np.arange((nrs-1)*kneighbors):
				cov[i,j] = ((M-1.)/M)* np.sum( (cdf_a[:,i] - np.average(cdf_a,axis=0)[i]) * (cdf_a[:,j] - np.average(cdf_a,axis=0)[j]), axis=0)
            
		return errors,cov     
        
        
# 	c2=np.empty((kneighbors,nrs,nrs))
# 	for a in range(kneighbors):
# 		cov2=np.empty((nrs,nrs))
# 		for b in range(kneighbors):
# 			cdf_a=cdf_arr[a::kneighbors]
# 			cdf_b=cdf_arr[b::kneighbors]
# 			cov = np.zeros(shape = (nrs,nrs))
# 			for i in np.arange(nrs):
# 				for j in np.arange(nrs):
# 					cov[i,j] = ((M-1.)/M)* np.sum( (cdf_a[:,i] - np.average(cdf_b,axis=0)[i]) * (cdf_a[:,j] - np.average(cdf_b,axis=0)[j]), axis=0)
# 			cov = np.delete(cov, -1, axis=0)
# 			cov = np.delete(cov, -1, axis=1)
# 			if b == 0:
# 				covar = cov
# 			else:
# 				covar = np.concatenate((covar,cov))
# 		if a==0:      
# 			covariance=covar
# 		else:
# 			covariance=np.concatenate((covariance,covar),axis=1)
              
# 	return errors,covariance



# 	c2=np.empty((kneighbors,nrs,nrs))
# 	for a in range(kneighbors):
# 		cov2=np.empty((nrs,nrs))
# 		for b in range(kneighbors):
# 			cdf_a=cdf_arr[a::kneighbors]
# 			cdf_b=cdf_arr[b::kneighbors]
# 			cov = np.zeros(shape = (nrs,nrs))
# 			for i in np.arange(nrs):
# 				for j in np.arange(nrs):
# 					cov[i,j] = ((M-1.)/M)* np.sum( (cdf_a[:,i] - np.average(cdf_b,axis=0)[i]) * (cdf_a[:,j] - np.average(cdf_b,axis=0)[j]), axis=0)
# 			if b == 0:
# 				covar = np.vstack((cov2[None],cov[None]))
# 			else:
# 				covar = np.vstack((covar,cov[None]))
# 		covar=covar[1:]
# 		if a==0:      
# 			covariance=np.concatenate((c2[None],covar[None]))
# 		else:
# 			covariance=np.concatenate((covariance,covar[None]))
# 	covariance=covariance[1:]
              
# 	return errors,covariance
    
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

def kNN_diff_error(d1,d2,d1err,d2err,single_neighbor=False):
	squared = np.vectorize(func)
	divide = np.vectorize(func2)
	squareroot = np.vectorize(func3)
	if single_neighbor==True:
		diff_error = np.zeros(len(d1))
		for i in range(len(d1)):
			diff_error[i] = math.sqrt(((d1err[i]**2)+(d2err[i])**2))
		return diff_error
	diff_error = np.zeros((len(d1),len(d1[0])))
	for i in range(len(d1)):                   
            diff_error[i]= squareroot((squared(d1err[i])+squared(d2err[i])))
	return diff_error


def kNN_chi_squared(d1,d2,d1cov,d2cov,single_neighbor=False,concatenate=False):
	cov = d1cov+d2cov
	if d1cov.shape != d2cov.shape:
		sys.exit("Error: arrays different lengths")
	if single_neighbor == True:
		invcov = np.linalg.inv(cov)
		diff = d1-d2
		return np.sum(np.dot(diff,np.dot(invcov,diff)))
	elif concatenate == False:
		chi_list =np.array([])
		for k in range(len(d1)):
			invcov = np.linalg.inv(cov[k])
			diff = d1[k]-d2[k]
			chi_list = np.append(chi_list,np.sum(np.dot(diff,np.dot(invcov,diff))))         
		return chi_list
	else:
		invcov = np.linalg.inv(cov)
		diff = (d1-d2).flatten()
		count = 0
		for i in range(len(cov)):
			for j in range(len(cov)):
				count += np.sum(np.dot(diff[j],np.dot(invcov[i][j],diff[i])))
		return count

def chop_cdf(rs,pis,high,low,errhigh,errlow,cov_high,cov_low,ind):
	inrangehigh = ((high[ind]>=0.1) & (high[ind]<=0.9))
	inrangelow = ((low[ind]>=0.1) & (low[ind]<=0.9))
	if(np.count_nonzero(inrangehigh == False) > np.count_nonzero(inrangelow == False)):
		inrange = inrangehigh
	else:
		inrange = inrangelow
	highnew = (high[ind])[inrange]
	lownew = (low[ind])[inrange]
    
	rsnew = rs[inrange]    
	pisnew = pis[inrange]

	errhighnew = errhigh[inrange]
	errlownew = errlow[inrange]
    
	indices = np.array(np.where(inrange==True))[0]

	cov_highnew = cov_high[ind]
	cov_lownew = cov_low[ind]
	for j in range(len(cov_highnew)-1-indices[-1]):
		cov_highnew = np.delete(cov_highnew, -1, axis=0)
		cov_highnew = np.delete(cov_highnew, -1, axis=1)
		cov_lownew = np.delete(cov_lownew, -1, axis=0)
		cov_lownew = np.delete(cov_lownew, -1, axis=1)
        
            
            
	for j in range(indices[0]):
		cov_highnew = np.delete(cov_highnew, 0, axis=0)
		cov_highnew = np.delete(cov_highnew, 0, axis=1)
		cov_lownew = np.delete(cov_lownew, 0, axis=0)
		cov_lownew = np.delete(cov_lownew, 0, axis=1)
        
	return rsnew, pisnew, highnew, lownew, errhighnew, errlownew, cov_highnew, cov_lownew

def angular_chop_cdf(angles,high,low,errhigh,errlow,cov_high,cov_low):
	inrangehigh = ((high>=0.05) & (high<=0.95))
	inrangelow = ((low>=0.05) & (low<=0.95))
	if(np.count_nonzero(inrangehigh == False) > np.count_nonzero(inrangelow == False)):
		inrange = inrangehigh
	else:
		inrange = inrangelow

# 	inrange = (((high[ind]>=0.05) & (high[ind]<=0.95)) & ((low[ind]>=0.05) & (low[ind]<=0.95)))
    
	highnew = (high)[inrange]
	lownew = (low)[inrange]
    
	anglesnew = angles[inrange]    

	errhighnew = errhigh[inrange]
	errlownew = errlow[inrange]
    
	indices = np.array(np.where(inrange==True))[0]

	cov_highnew = cov_high
	cov_lownew = cov_low
	for j in range(len(cov_highnew)-1-indices[-1]):
		cov_highnew = np.delete(cov_highnew, -1, axis=0)
		cov_highnew = np.delete(cov_highnew, -1, axis=1)
		cov_lownew = np.delete(cov_lownew, -1, axis=0)
		cov_lownew = np.delete(cov_lownew, -1, axis=1)
        
                
	for j in range(indices[0]):
		cov_highnew = np.delete(cov_highnew, 0, axis=0)
		cov_highnew = np.delete(cov_highnew, 0, axis=1)
		cov_lownew = np.delete(cov_lownew, 0, axis=0)
		cov_lownew = np.delete(cov_lownew, 0, axis=1)
        
	return anglesnew, highnew, lownew, errhighnew, errlownew, cov_highnew, cov_lownew
    
    
    
    
    
    
    
    
    
    
    