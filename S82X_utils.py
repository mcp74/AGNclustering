import numpy as np 
from numpy.lib.recfunctions import append_fields
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
from astropy import units as u
import sys
from scipy import interpolate
from pathlib import Path


def get_flux_f(data_path,waveband='full'):
    s82area=20.21

    if waveband=='full':
        fluxareaf=np.genfromtxt(data_path+'flux-area/xmm_ao10_ao13_af_0.5-10keV.txt')
        F = fluxareaf[:,0]
        A = fluxareaf[:,1]
        f = interpolate.interp1d(Ff,Af)

    if waveband=='hard':
        fluxareaf=np.genfromtxt(data_path+'flux-area/xmm_ao10_ao13_af_2-10keV.txt')
        F = fluxareaf[:,0]
        A = fluxareaf[:,1]
        f = interpolate.interp1d(Ff,Af)

    else: f=None
    return f



def area_hard(data_path,log_flux):
    if log_flux>-13.2:
        return s82area
    elif log_flux<-13.85:
        return 1e-2
    else:
        fh = get_flux_f(data_path,waveband='hard')
        return fh(log_flux)

def area_full(data_path,log_flux):
    if log_flux>-13.3:
        return s82area
    elif log_flux<-14.78:
        return 1e-2
    else:
        ff = get_flux_f(data_path,waveband='full')
        return ff(log_flux)

def include_area_weights(data_path,cat,band):
    p_arr=[]
    if band=='full':
        flux=cat['flux_full']
        fn = area_full
        for f in flux:
            p = area_full(data_path,np.log10(f))/s82area
            p_arr.append(1./p)
    elif band=='hard':
        flux=cat['flux_hard']
        fn = area_hard
        for f in flux:
            p = area_hard(data_path,np.log10(f))/s82area
            p_arr.append(1./p)
    #for f in cat['flux']:
    #    p = area(np.log10(f))/s82area
    #    p_arr.append(1./p)
    norm = len(cat)/np.sum(p_arr)
    w_arr=norm*np.array(p_arr)
    if 'weight' in cat.dtype.names:
        w_arr = w_arr * cat['weight']
    temp = list(zip(cat['z'],cat['ra'],cat['dec'],flux,w_arr))
    new = np.zeros((len(cat),), dtype=[('z', '<f8'),('ra', '<f8'),('dec', '<f8'),('flux', '<f8'),('weight', '<f8')])
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

def parseS82X_pdfs(data, pdf_dir=None, prefix='Id', extension='spec', zfill=9):
    '''
    Input: data array with following column names:
    ra: 'CP_RA'
    dec: 'CP_DEC'
    spec-z: 'SPEC_Z'
    phot-z: 'PHOTO_Z'
    'PDZ'

    returns parsed array with z weights, array of PDFs
    '''

    w_arr=[]
    z_arr=[]
    ra_arr=[]
    dec_arr=[]
    id_arr=[]
    Hflux_arr=[]
    Fflux_arr=[]
    ztype=[]
    pdfs=[]
    skip=0

    for i,r in enumerate(data):
        if (r['SPEC_Z']<=0) and (r['PHOTO_Z']>0):
            if pdf_dir is None:
                sys.exit('Please specify directory for photo-z pdf files')

            fname = pdf_dir+prefix+str(r['REC_NO']).zfill(zfill)+'.'+extension
            try:
                full_pdf = np.genfromtxt(fname,skip_header=32)[:651]
            except IOError:
                #print('unable to find PDF file')
                skip = skip+1
                continue

            pdf=full_pdf[np.where(full_pdf[:,1]>1e-4)]
            p=pdf[:,1]
            z=pdf[:,0]
            #if len(p)>100:
            #   p=p[::10]
            #   z=z[::10]
            p=p/sum(p)
            ra = r['CP_RA']*np.ones(len(p))
            dec = r['CP_DEC']*np.ones(len(p))
            ident = r['REC_NO']*np.ones(len(p))
            Hflux = r['HARD_FLUX']*np.ones(len(p))
            Fflux = r['FULL_FLUX']*np.ones(len(p))
            phot = np.ones(len(p))
            w_arr=np.append(w_arr,p)
            z_arr=np.append(z_arr,z)
            ra_arr=np.append(ra_arr,ra)
            dec_arr=np.append(dec_arr,dec)
            Hflux_arr=np.append(Hflux_arr,Hflux)
            Fflux_arr=np.append(Fflux_arr,Fflux)
            id_arr=np.append(id_arr,ident)
            ztype=np.append(ztype,phot)
            pdfs.append([z,p])
        else:
            w_arr=np.append(w_arr,1.)
            z_arr=np.append(z_arr,r['SPEC_Z'])
            ra_arr=np.append(ra_arr,r['CP_RA'])
            dec_arr=np.append(dec_arr,r['CP_DEC'])
            id_arr=np.append(id_arr,r['REC_NO'])
            Hflux_arr=np.append(Hflux_arr,r['HARD_FLUX'])
            Fflux_arr=np.append(Fflux_arr,r['FULL_FLUX'])
            ztype=np.append(ztype,0)
            pdfs.append([np.array([r['SPEC_Z']]),np.array([1.])])
    print('Could not find '+str(skip)+' PDF files.')
    temp = list(zip(ra_arr,dec_arr,z_arr,Hflux_arr,Fflux_arr,w_arr,id_arr,ztype))
    parsed = np.zeros((len(ra_arr),), dtype=[('ra', '<f8'),('dec', '<f8'),('z', '<f8'),('flux_hard', '<f8'),('flux_full', '<f8'),('weight', '<f8'),('id', '<f8'),('ztype', '<i8')])
    parsed[:] = temp

    return parsed, np.array(pdfs)

