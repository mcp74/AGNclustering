import numpy as np 
from numpy.lib.recfunctions import append_fields
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
from astropy import units as u
import sys

# New function to sum along rp
def sum_rp(xi):
	zp = np.zeros(40)
	k=0
	for i in range(10):     
		for j in range(40):
			zp[j] = zp[j] + 2.0*xi[j+k]
		k+=40
	return zp