from __future__ import division

import numpy as np 
from numpy.lib.recfunctions import append_fields
from astropy.table import Table, vstack
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
from astropy import units as u
import sys
from joblib import Parallel, delayed
from scipy.spatial import Voronoi
from scipy.cluster.vq import kmeans, vq

from AGNclustering.utils import *
from Corrfunc.mocks import DDrppi_mocks

def auto_jackknife(d,r,m,pimax,bins,estimator,covariance=True,survey=None):
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
        print('Performing jackknifes using k-means blocks')
        dcoords=np.vstack((d['ra'], d['dec'])).T
        rcoords=np.vstack((r['ra'], r['dec'])).T
        
        nclusters = m
        centers = kmeans(d, nclusters)[0]
        cluster = vq(d, centers)[0]
        vor = Voronoi(centers)

        dblocks = _vor_block(d,vor)
        rblocks = _vor_block(r,vor)

        return
    M=len(dblocks)
    print('Using ',M,' jacknife samples')
    nbins = len(bins)-1
    
    #NON-PARALLELIZED JACKKNIFE:
    wp_arr = np.empty(nbins)
    for i in np.arange(M):
        print(str(i+1),'/',str(M))
        new_dblocks = np.delete(dblocks,i,0)
        new_rblocks = np.delete(rblocks,i,0)
        dset = np.concatenate(new_dblocks)
        rset = np.concatenate(new_rblocks)
        wp_t = wp_dd(data=dset,randoms=rset,pimax=pimax,bins=bins,estimator=estimator)
        wp_arr=np.vstack((wp_arr,wp_t))
    wp_arr=wp_arr[1:]
    
    #PARALLELIZED JACKKNIFE COMPUTATION:
    #wp_arr = np.array(Parallel(n_jobs=-1)(delayed(PAjackknife)(i,dblocks,rblocks,pimax=pimax,bins=bins,estimator=estimator) for i in np.arange(M)))

    err =np.sqrt(((M-1.)/M) * np.sum( (wp_arr - np.average(wp_arr,axis=0))**2, axis=0))
    if covariance:
        cov = np.zeros(shape = (nbins,nbins))
        for i in np.arange(nbins):
            for j in np.arange(nbins):
                cov[i,j] = ((M-1.)/M)* np.sum( (wp_arr[:,i] - np.average(wp_arr,axis=0)[i]) * (wp_arr[:,j] - np.average(wp_arr,axis=0)[j]), axis=0)
        return err,cov
    else: return err

def cross_jackknife(d1,d2,r1,r2,m,pimax,bins,estimator,covariance=True,survey=None,weights1=None,weights2=None):
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
        print('Performing jackknifes using k-means blocks')
        d1coords=np.vstack((d1['ra'], d1['dec'])).T
        d2coords=np.vstack((d2['ra'], d2['dec'])).T
        if r1 is not None:
            r1coords=np.vstack((r1['ra'], r1['dec'])).T
        if r2 is not None:
            r2coords=np.vstack((r2['ra'], r2['dec'])).T

        if len(d2)>len(d1):
                data=np.vstack((d2['ra'], d2['dec'])).T
        else:
            data=np.vstack((d1['ra'], d1['dec'])).T
        
        nclusters = m
        centers = kmeans(data, nclusters)[0]
        cluster = vq(data, centers)[0]
        vor = Voronoi(centers)

        d1blocks = _vor_block(d1,vor)
        d2blocks = _vor_block(d2,vor)
        r1blocks = _vor_block(r1,vor)
        r2blocks = _vor_block(r2,vor)
        #return
    
    M=len(d1blocks)
    print('Using ',M,' jacknife samples')
    nbins = len(bins)-1

    #NON-PARALLELIZED JACKKNIFE:
    wp_arr = np.empty(nbins)
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
        wp_t = wp_d1d2(d1=d1set, d2=d2set, r1=r1set, r2=r2set, bins=bins, pimax=pimax, estimator=estimator)
        wp_arr=np.vstack((wp_arr,wp_t))
    wp_arr=wp_arr[1:]

    #PARALLELIZED JACKKNIFE COMPUTATION:
    #wp_arr = np.array(Parallel(n_jobs=-1)(delayed(PCjackknife)(i,d1blocks,d2blocks,r2blocks,pimax=pimax,bins=bins,estimator=estimator,r1blocks=r1blocks) for i in np.arange(M)))

    err =np.sqrt(((M-1.)/M) * np.sum( (wp_arr - np.average(wp_arr,axis=0))**2, axis=0))
    if covariance:
        cov = np.zeros(shape = (nbins,nbins))
        for i in np.arange(nbins):
            for j in np.arange(nbins):
                cov[i,j] = ((M-1.)/M)* np.sum( (wp_arr[:,i] - np.average(wp_arr,axis=0)[i]) * (wp_arr[:,j] - np.average(wp_arr,axis=0)[j]), axis=0)
        return err,cov
    else: return err

def PAjackknife(i,dblocks,rblocks,pimax,bins,estimator):
    new_dblocks = np.delete(dblocks,i,0)
    new_rblocks = np.delete(rblocks,i,0)
    dset = np.concatenate(new_dblocks)
    rset = np.concatenate(new_rblocks)
    wp_t = wp_dd(data=dset,randoms=rset,pimax=pimax,bins=bins,estimator=estimator)
    return wp_t

def PCjackknife(i,d1blocks,d2blocks,r2blocks,pimax,bins,estimator,r1blocks=None):
    new_d1blocks = np.delete(d1blocks,i,0)
    new_d2blocks = np.delete(d2blocks,i,0)

    if r1blocks != 0:
        new_r1blocks = np.delete(r1blocks,i,0)
    new_r2blocks = np.delete(r2blocks,i,0)
    d1set = np.concatenate(new_d1blocks)
    d2set = np.concatenate(new_d2blocks)
    if r1blocks != 0:
        r1set = np.concatenate(new_r1blocks)
    else:
        r1set=None
    r2set = np.concatenate(new_r2blocks)
    wp_t = wp_d1d2(d1=d1set, d2=d2set, r1=r1set, r2=r2set, bins=bins, pimax=pimax, estimator=estimator)
    return wp_t

def _BASS_block(cat,m):

    if cat is None:
        return 0
    blocks=[]
    lens=[]

    ra0=0
    ra_int=(360./m)
    dec_int=(180./m)
    #dec_int=(2./m)
    for i in range(m):
        dec0=-90
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
    return np.asarray(blocks, dtype="object")

def _AO13_block(cat,m):
    if cat is None:
        return 0
    blocks=[]
    lens=[]

    ramin=14.1
    ramax=28.04
    ra_int=((ramax-ramin)/m)
    ra0=ramin
    dec_min=-0.63
    dec_max=0.63
    for i in range(m):
        ra_min=ra0
        ra_max=ra0+ra_int
        block=cat[(cat['ra']<ra_max)&(cat['ra']>=ra_min)&\
                (cat['dec']<dec_max)&(cat['dec']>=dec_min)]
        dec0=dec_max
        blocks.append(block)
        lens.append(len(block))
        #print ra_min,dec_min
        ra0=ra_max
    return np.asarray(blocks, dtype="object")

def _S82X_block(cat, m):
    if cat is None:
        return 0
    blocks=[]
    blocks.append(cat[(cat['ra']>300)&(cat['ra']<=332)])
    blocks.append(cat[(cat['ra']>332)&(cat['ra']<=334.3152435687774)])
    blocks.append(cat[(cat['ra']>334.3152435687774)&(cat['ra']<=340)])
    blocks.append(cat[(cat['ra']>340)&(cat['ra']<=352.50688215)])
    blocks.append(cat[(cat['ra']>352.50688215)&(cat['ra']<=353.5)])
    blocks.append(cat[(cat['ra']>353.5)&(cat['ra']<=358)])
    blocks.append(cat[(cat['ra']>358)|(cat['ra']<=4)])
    blocks.append(cat[(cat['ra']>4)&(cat['ra']<14.1)])
    ramin=14.1
    ramax=28.04
    ra_int=((ramax-ramin)/10)
    ra0=ramin
    dec_min=-0.63
    dec_max=0.63
    for i in range(10):
        ra_min=ra0
        ra_max=ra0+ra_int
        block=cat[(cat['ra']<ra_max)&(cat['ra']>=ra_min)]
        dec0=dec_max
        blocks.append(block)
        ra0=ra_max
    blocks.append(cat[(cat['ra']>28.04)&(cat['ra']<=42)])
    blocks.append(cat[(cat['ra']>42)&(cat['ra']<=50)])
    blocks.append(cat[(cat['ra']>50)&(cat['ra']<=60)])
    return np.asarray(blocks, dtype="object")

def _XXL_block(cat, m=None):
    if cat is None:
        return 0
    perct=False #to print # per jackknife sample
    blocks=[]
    
    #XMM-XXL
    ramin=30.0592
    ramax=38.8278
    decmin=-7.34838
    decmax=-2.67178
    n=5
    ra_int=((ramax-ramin)/n)
    dec_int=((decmax-decmin)/n)
    dec0=decmin
    patch=[]
    for i in range(n):
        dec_min=dec0
        dec_max=dec0+dec_int
        ra0=ramin
        for j in range(n):
            ra_min=ra0
            ra_max=ra0+ra_int
            new=cat[(cat['ra']<ra_max)&(cat['ra']>=ra_min)&(cat['dec']<dec_max)&(cat['dec']>=dec_min)]
            if len(patch)==0:
                patch=new
            else:
                patch=np.array(vstack([Table(patch),Table(new)]))
            if (((i==1)&(j==1))|((i==1)&(j==4))|((i==2)&(j>0))|((i==3)&(j>0)&(j<4))|((i==4)&(j==3))):
                blocks.append(patch)
                patch=[]
            elif ((i==3)&(j==4)):
                side=patch
                patch=[]
            elif((i==4)&(j==4)):
                patch=np.array(vstack([Table(patch),Table(side)]))
                blocks.append(patch)
                patch=[]
            ra0=ra_max
        dec0=dec_max

    if perct:
        for i, b in enumerate(blocks):
            if 'weight' in cat.dtype.names:
                print(i, (sum(b['weight'])/sum(cat['weight']))*100)
            else:
                print(i, (len(b)/len(cat))*100)
    return np.asarray(blocks, dtype="object")


def _Comb_block(cat, m=None):
    if cat is None:
        return 0
    perct=False
    blocks=[]

    #AO10 - 1
    blocks.append(cat[(cat['ra']>332)&(cat['ra']<=334.3152435687774)&(cat['dec']<=1.5)&(cat['dec']>=-1.5)])
    blocks.append(cat[(cat['ra']>334.3152435687774)&(cat['ra']<=335)&(cat['dec']<=1.5)&(cat['dec']>=-1.5)])

    #AO10 - 2
    blocks.append(cat[(cat['ra']>351)&(cat['ra']<=352.50688215)&(cat['dec']<=1.5)&(cat['dec']>=-1.5)])
    blocks.append(cat[(cat['ra']>352.50688215)&(cat['ra']<=353.5)&(cat['dec']<=1.5)&(cat['dec']>=-1.5)])


    #AO13
    ramin=14.1
    ramax=28.04
    ra_int=((ramax-ramin)/10)
    ra0=ramin
    for i in range(10):
    #for i in range(15):
        ra_min=ra0
        ra_max=ra0+ra_int
        block=cat[(cat['ra']<ra_max)&(cat['ra']>=ra_min)&(cat['dec']<=1.5)&(cat['dec']>=-1.5)]
        blocks.append(block)
        ra0=ra_max
    
    
    #XMM-XXL
    ramin=30.0592
    ramax=38.8278
    decmin=-7.34838
    decmax=-2.67178
    n=5
    ra_int=((ramax-ramin)/n)
    dec_int=((decmax-decmin)/n)
    dec0=decmin
    patch=[]
    for i in range(n):
        dec_min=dec0
        dec_max=dec0+dec_int
        ra0=ramin
        for j in range(n):
            ra_min=ra0
            ra_max=ra0+ra_int
            new=cat[(cat['ra']<ra_max)&(cat['ra']>=ra_min)&(cat['dec']<dec_max)&(cat['dec']>=dec_min)]
            if len(patch)==0:
                patch=new
            else:
                patch=np.array(vstack([Table(patch),Table(new)]))
            if (((i==1)&(j==1))|((i==1)&(j==4))|((i==2)&(j>0))|((i==3)&(j>0)&(j<4))|((i==4)&(j==3))):
                blocks.append(patch)
                patch=[]
            elif ((i==3)&(j==4)):
                side=patch
                patch=[]
            elif((i==4)&(j==4)):
                patch=np.array(vstack([Table(patch),Table(side)]))
                blocks.append(patch)
                patch=[]
            ra0=ra_max
        dec0=dec_max

    if perct:
        for i, b in enumerate(blocks):
            if 'weight' in cat.dtype.names:
                print(i, (sum(b['weight'])/sum(cat['weight']))*100)
            else:
                print(i, (len(b)/len(cat))*100)
    return np.asarray(blocks, dtype="object")


def _vor_block(cat,vor):
    if cat is None:
        return 0
    indices=[]
    blocks=[]
    coords=np.vstack((cat['ra'], cat['dec'])).T
    for coord in coords:
        point_index = np.argmin(np.sum((vor.points - coord)**2, axis=1))
        indices.append(point_index)
    cat = append_fields(cat, 'vbin', np.array(indices))

    for i in range(len(vor.points)):
        block=cat[(cat['vbin']==i)]
        blocks.append(block)
    return np.asarray(blocks, dtype="object")