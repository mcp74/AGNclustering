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

from AGNclustering.utils import z_to_cdist
from AGNclustering.kNN_error import kNN_jacknife
from AGNclustering.kNN_error import angular_kNN_jacknife
from AGNclustering.KNN_stuff import CDFkNN_rp_pi
from AGNclustering.angular_kNN import CDFkNN_theta


def kNN_wrapper(rs, pis, agn, gal, rslist, rlistbool, m, kneighbors,rpbool=True,concatenate=False):
    
	cdf=CDFkNN_rp_pi(rs,pis, agn, gal,kneighbors=kneighbors)
	err, cov = kNN_jacknife(d=agn,r=gal,rs=rs,pis=pis,kneighbors=kneighbors,rslist=rslist,rlistbool=rlistbool,m=m,rpbool=rpbool,concatenate=concatenate)
	return cdf, err, cov

def angular_kNN_wrapper(angles, agn, gal, kneighbors, angleslist, m, concatenate = False):
    
	decg = agn['dec'] * ((np.pi)/180.0)
	rag = agn['ra'] * ((np.pi)/180.0)

	decr = gal['dec'] * ((2*np.pi)/360.0)
	rar = gal['ra'] * ((2*np.pi)/360.0)

	agn_angles = np.vstack((decg, rag)).T
	gal_angles = np.vstack((decr, rar)).T

	outputs = CDFkNN_theta(angles, gal_angles, agn_angles, kneighbors)
    
	err, cov = angular_kNN_jacknife(d=agn, r=gal, angles=angles, kneighbors=kneighbors, angleslist=angleslist, concatenate=concatenate, m=m)
    
	return outputs, err, cov
    
    
    
    
    
