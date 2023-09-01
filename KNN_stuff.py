import numpy as np
import scipy.spatial
import sys, time
from functools import partial

import ray
from joblib import Parallel, delayed

from astropy import units as u
import astropy.coordinates as coord
import math as m

from itertools import repeat
import multiprocessing
from multiprocessing import Pool

from fast_histogram import histogram2d

from AGNclustering.utils import z_to_cdist

import pyfnntw


def is_linear_or_log(X):
    
	diffs = np.diff(X)
	ratios = np.divide(diffs[:-1], diffs[1:])
	avg_diff = np.mean(diffs)
	avg_ratio = np.mean(ratios)
	std_diff = np.std(diffs)
	std_ratio = np.std(ratios)

	if abs(std_diff) < 1e-4:
		return "linear"
	elif abs(std_ratio) < 1e-4:
		return "log"
	else:
		return "neither"

def get_trans_par_fnntw(data, query, k, LS = 32, lbox = 1):
	start = time.time()
	if lbox <= 0:
		xtree = pyfnntw.Treef32(data, leafsize=LS)
	else:
		lbox = np.float32(lbox)
		xtree = pyfnntw.Treef32(data, leafsize=LS, boxsize = np.array([lbox, lbox, lbox]))
	print("build tree", time.time()-start)


	start = time.time()
# 	par, trans = xtree.query(query, k=k[-1], axis=2)
	par, trans = xtree.query(query, k=k, axis=2)
	print("query", time.time()-start)


	return trans, par


def calc_cdf_hist(rs, pis, dis_t, dis_p):

	cdfs = np.zeros((dis_t.shape[1], len(rs), len(pis)), dtype = np.float32)

    # are the bins lin or log
	rpbins = np.concatenate((np.zeros(1, dtype = np.float32), rs))
	pibins = np.concatenate((np.zeros(1, dtype = np.float32), pis))
	scaling_t = is_linear_or_log(rpbins)
	scaling_p = is_linear_or_log(pibins)

	assert scaling_t == scaling_p

	start = time.time()
	args_list = [(ik, dis_t, dis_p, rpbins, pibins, scaling_t) for ik in range(dis_t.shape[1])]
	results = ray.get([calculate_cdfs.remote(*args) for args in args_list])
	# results = Parallel(n_jobs=dis_t.shape[1])(delayed(calculate_cdfs)(*args) for args in args_list)
	print("cdf parallel", time.time() - start)


	for ik, cdf in results:
		cdfs[ik] = cdf 
	return cdfs

@ray.remote
def calculate_cdfs(ik, dis_t, dis_p, rpbins, pibins, scaling):
	if scaling == 'linear':
		dist_hist2d_k = histogram2d(dis_t[:, ik], dis_p[:, ik],range=[[rpbins[0], rpbins[-1]], [pibins[0], pibins[-1]]], bins=[len(rpbins)-1, len(pibins)-1])
	else:
		dist_hist2d_k, _, _ = np.histogram2d(dis_t[:, ik], dis_p[:, ik], bins=(rpbins, pibins))
      
	dist_cdf2d_k = np.cumsum(np.cumsum(dist_hist2d_k, axis=0), axis=1)
	cdf = dist_cdf2d_k / dist_cdf2d_k[-1, -1]
	return (ik, cdf)

def CDFkNN_rp_pi(rs, pis, xgalfull, xrandfull, kneighbors = 1, nthread = 32, periodic = 0, randdown = 1, LS = 32):
  
	xgal = convert_to_cartesian(xgalfull)
	xrand = convert_to_cartesian(xrandfull)
    
	assert xgal.shape[1] == 3
	assert xrand.shape[1] == 3

   

	rs = np.float32(rs)
	pis = np.float32(pis)
	xgal = np.float32(xgal)
	xrand = np.float32(xrand)
  
	xgal = np.array(xgal, order="C")
	xrand = np.array(xrand, order="C")

	if randdown > 1:
		xrand = xrand[np.random.choice(np.arange(len(xrand)), size = int(len(xrand)/randdown), replace = False)]
	print("Ngal", len(xgal), "Nrand", len(xrand), kneighbors)
   
	start = time.time()
 
	dis_t, dis_p = get_trans_par_fnntw(xgal, xrand, kneighbors, lbox = periodic, LS = LS)
    
	print("  kdtree tot", time.time() - start) # 1.6 seconds, 1 second of which was sorting
 
	assert dis_t.shape == dis_p.shape

 
	start = time.time()
	outputs1 = calc_cdf_hist(rs, pis, dis_t, dis_p)
	print("  cdf", time.time() - start) # 1.6 seconds, 1 second of which was sorting

	return outputs1

def convert_to_cartesian(agn):
	if 'cdist' not in agn.dtype.names:
		agn = z_to_cdist(agn)
	c = agn['cdist']
	a = coord.Angle(agn['ra']*u.degree)
	b = coord.Angle(agn['dec']*u.degree)
	agnspatial = np.empty((1,3), dtype=float)
	for i in range(len(agn)):
		x = c[i] * m.cos(b[i].radian) * m.cos(a[i].radian)
		y = c[i] * m.cos(b[i].radian) * m.sin(a[i].radian)
		z = c[i] * m.sin(b[i].radian)
		agnspatial = np.vstack((agnspatial,[x,y,z]))
	agnspatial = np.delete(agnspatial, 0, axis=0)
	return agnspatial
    