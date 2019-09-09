import numpy as np 
from numpy.lib.recfunctions import append_fields
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
from astropy import units as u
import sys
from scipy import interpolate

path = '~/Dropbox/Projects/clustering/XXL/'
#fluxarea=np.genfromtxt(path+'XMM_XXL_tot_af_0.5-10keV.txt', delimiter=',')


def parse(band):
    if band=='full':
        #fluxarea=np.genfromtxt(path+'FvA_0.5-10keV.csv',delimiter=',')
        fluxarea=np.genfromtxt(path+'flux_area/N_full_40.curve')
    elif band=='hard':
        #fluxarea=np.genfromtxt(path+'FvA_2-10keV.csv',delimiter=',')
        fluxarea=np.genfromtxt(path+'flux_area/N_hard_40.curve')
    F = -fluxarea[:,1]
    A = fluxarea[:,4]
    #F = fluxarea[:,0]
    #A = fluxarea[:,1]
    fn = interpolate.interp1d(F,A)
    return fn

def area(flux,fn):
    lflux = np.log10(flux)
    if lflux > -12.1:
        return fn(-12)
    elif lflux < -15:
        return 1e-4
    else:
        return fn(lflux)

def include_area_weights(cat,band='full'):
    p_arr=[]
    if band=='full':
        flux=cat['flux_full']
        fn = parse('full')
    elif band=='hard':
        flux=cat['flux_hard']
        fn = parse('hard')
    Xarea=fn(-12)
    for f in flux:
        p = area(f,fn)/Xarea
        p_arr.append(1./p)
    norm = len(cat)/np.sum(p_arr)
    w_arr=norm*np.array(p_arr)
    if 'weight' in cat.dtype.names:
        w_arr = w_arr * cat['weight']
    temp = list(zip(cat['z'],cat['nh'],cat['ra'],cat['dec'],flux,w_arr))
    new = np.zeros((len(cat),), dtype=[('z', '<f8'),('nh', '<i8'),('ra', '<f8'),('dec', '<f8'),('flux', '<f8'),('weight', '<f8')])
    
    new[:] = temp
    return new

def standardize(cat):
    if 'weight' in cat.dtype.names:
        w_arr = cat['weight']
    else: w_arr = np.ones(len(cat))
    temp = list(zip(cat['z'],cat['ra'],cat['dec'],cat['flux_full'],w_arr))
    new = np.zeros((len(cat),), dtype=[('z', '<f8'),('ra', '<f8'),('dec', '<f8'),('flux', '<f8'),('weight', '<f8')])
    new[:] = temp
    return new

def boss_footprint(data=None,ra=None,dec=None):
    if hasattr(data, "__len__"):
        if 'ra' in data.dtype.names:
            ra=data['ra']
            dec=data['dec']
        if 'RA' in data.dtype.names:
            ra=data['RA']
            dec=data['DEC']
    plate_r = 1.5
    ra1 = 31.5
    dec1= -5.6
    ra2 = 33.7
    dec2= -4.9
    ra3 = 35.4
    dec3= -4.615
    ra4 = 35.88
    dec4= -4.27
    ra5 = 37.35
    dec5= -4.8
    r1 = np.sqrt( (ra-ra1)**2 + (dec-dec1)**2 )
    r2 = np.sqrt( (ra-ra2)**2 + (dec-dec2)**2 )
    r3 = np.sqrt( (ra-ra3)**2 + (dec-dec3)**2 )
    r4 = np.sqrt( (ra-ra4)**2 + (dec-dec4)**2 )
    r5 = np.sqrt( (ra-ra5)**2 + (dec-dec5)**2 )
    if hasattr(data, "__len__"):
        cond = (r1<=plate_r) | (r2<=plate_r) | (r3<=plate_r) | (r4<=plate_r) | (r5<=plate_r)
        return data[cond]
    else:
        if (r1<=plate_r) | (r2<=plate_r) | (r3<=plate_r) | (r4<=plate_r) | (r5<=plate_r):
            return True
        else:
            return False
    

