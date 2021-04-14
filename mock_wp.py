import numpy as np
import math
import time
from astropy import units as u
from astropy import constants
from astropy.cosmology import Planck15,FlatLambdaCDM
from astropy.table import Table
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from AbundanceMatching import *
from halotools.sim_manager import CachedHaloCatalog
#from halotools.empirical_models import PrebuiltHodModelFactory, PrebuiltSubhaloModelFactory
from halotools.mock_observables import return_xyz_formatted_array
from halotools.mock_observables import wp
from scipy import interpolate


def wpgg_mock(rpbins,halos,Mabs_arr,Vsim_Vsur_ratio,Lbox,pi_max=40,):
	'''
	Selects mock galaxies with same K mag distribution as survey, and computes its autocorrelation function.
	Requires halotools.mock_observables
	
	rpbins: bins of projected separation
	halos: catalog of dm halos with absolute K-band magnitude assigned 'Kmag'
	Mabs_arr: array of K-band magnitudes from survey
	Vsim_Vsur_ratio: ratio of simulation volume to survey volume
	pi_max: default 40 Mpc/h
	Lbox: size of simulation box
	'''

    rpwidths=[]
    for i in np.arange(len(rpbins)-1):
        rpwidths.append(rpbins[i+1] - rpbins[i])
    rp = rpbins[1:] - 0.5*np.array(rpwidths)
    
    bins = np.linspace(-27.1,-19,50)
    mhist,mbins = np.histogram(halos['Kmag'],bins)
    dhist,dbins = np.histogram(Mabs_arr,bins)

    f = interpolate.interp1d((bins[1:]-delt),(dhist/mhist)*Vsim_Vsur_ratio)
    try:
        f_arr = np.where(halos['Kmag']<-20.5, f(halos['Kmag']),0)
    except ValueError:
        print(max(bins[1:]-delt),max(halos['Kmag']),min(bins[1:]-delt),min(halos['Kmag']))
    rand_arr = np.random.rand(len(halos))
    
    good = (rand_arr<f_arr)
    x = halos['halo_x'][good]
    y = halos['halo_y'][good]
    z = halos['halo_z'][good]
    all_positions = return_xyz_formatted_array(x, y, z)
    print(len(x))

    try:
        return all_positions,rp,wp(all_positions, rpbins, pi_max, period=Lbox, num_threads=4)
    except ValueError:
        print('Not enough')


def chi_square(rp,data,model,cov_matrix):
    diff = data-model
    invcov = np.linalg.inv(cov_matrix)
    return np.sum(np.dot(diff,np.dot(invcov,diff)))


def auto_jackknife(cat,m,Lbox,rpbins,covariance=True,pi_max=40):
	'''
    Computes errors due to cosmic variance on mock correlation function via jackknife resampling.
    
    cat: catalog of mock galaxy positions
    m: sqrt(M) where M is total number of jackknife samples
    Lbox: size of simulation box
    rpbins: projected separation bins
	'''

    gblocks=_block(cat,m,Lbox)

    M=len(gblocks)
    nbins = len(rpbins)-1
    wp_arr = np.empty(nbins)
    
    #compute wp without one block
    for i in np.arange(M):
        new_gblocks = np.delete(gblocks,i,0)
        gset = np.concatenate(new_gblocks)
        x = gset['x']
        y = gset['y']
        z = gset['z']
        gset_p = return_xyz_formatted_array(x, y, z)
        wp_t=wp(gset_p, rpbins, pi_max,period=Lbox, num_threads=4)
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


def _block(cat,m,Lbox):
    blocks=[]
    x0=0
    delt = Lbox[0]/m
    for i in range(m):
        y0=0
        for j in range(m):
            z0=0
            for k in range(m):
                x_min=x0
                x_max=x0+delt
                y_min=y0
                y_max=y0+delt
                z_min=z0
                z_max=z0+delt
                block=cat[(cat['x']<x_max)&(cat['x']>=x_min)&\
                          (cat['y']<y_max)&(cat['y']>=y_min)&\
                          (cat['z']<z_max)&(cat['z']>=z_min)]
                #x0=x_max
                #y0=y_max
                z0=z_max
                blocks.append(block)
                #print ra_min,dec_min
            y0=y_max
        x0=x_max
    return blocks
