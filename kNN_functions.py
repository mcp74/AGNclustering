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

from numba import njit, types, jit, prange
import numba
from fast_histogram import histogram2d

from AGNclustering.utils import z_to_cdist
from AGNclustering.kNN_error import kNN_jacknife
from AGNclustering.KNN_stuff import CDFkNN_rp_pi


def kNN_wrapper(rs, pis, agn, gal, m, kneighbors,rpbool=True):
    
	cdf=CDFkNN_rp_pi(rs,pis, agn, gal,kneighbors=kneighbors)
	err, cov = kNN_jacknife(d=agn,r=gal,rs=rs,pis=pis,kneighbors=kneighbors,m=m,rpbool=rpbool)
	return cdf, err, cov
