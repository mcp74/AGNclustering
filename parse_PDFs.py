from __future__ import division

import numpy as np 
from numpy.lib.recfunctions import append_fields
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import sys


def parseS82X_pdfs(data, pdf_dir, prefix='Id', extension='spec', zfill=9):
	'''
	Input: data array with following column names:
	ra: 'CP_RA'
	dec: 'CP_DEC'
	spec-z: 'SPEC_Z'
	phot-z: 'PHOTO_Z'
	'PDZ'

	returns parsed array with z weights, array of PDFs
	'''
	if pdf_dir is None:
		sys.exit('Please specify directory for photo-z pdf files')

	w_arr=[]
	z_arr=[]
	ra_arr=[]
	dec_arr=[]
	id_arr=[]
	flux_arr=[]
	ztype=[]
	pdfs=[]
	skip=0
	for i,r in enumerate(data):
		if (r['SPEC_Z']<=0) and (r['PHOTO_Z']>0):
			fname = pdf_dir+prefix+str(r['REC_NO']).zfill(zfill)+'.'+extension
			try:
				full_pdf = np.genfromtxt(fname,skip_header=32)[:651]
			except IOError:
				print('unable to find PDF file')
				skip = skip+1
				continue

			pdf=full_pdf[np.where(full_pdf[:,1]>1e-4)]
			p=pdf[:,1]
			z=pdf[:,0]
			#if len(p)>100:
			#	p=p[::10]
			#	z=z[::10]
			p=p/sum(p)
			ra = r['CP_RA']*np.ones(len(p))
			dec = r['CP_DEC']*np.ones(len(p))
			ident = r['REC_NO']*np.ones(len(p))
			flux = r['FULL_FLUX']*np.ones(len(p))
			phot = np.ones(len(p))
			w_arr=np.append(w_arr,p)
			z_arr=np.append(z_arr,z)
			ra_arr=np.append(ra_arr,ra)
			dec_arr=np.append(dec_arr,dec)
			flux_arr=np.append(flux_arr,flux)
			id_arr=np.append(id_arr,ident)
			ztype=np.append(ztype,phot)
			#Because dz is 0.02 for z>6:
			#lz = (z<=6)
			#hz = (z>6)
			#new1=p[lz]
			#new2=.5*np.repeat(p[hz],2.)
			#newp=np.append(new1,new2)
			#pdfs.append(newp)
			pdfs.append([z,p])
		else:
			w_arr=np.append(w_arr,1.)
			z_arr=np.append(z_arr,r['SPEC_Z'])
			ra_arr=np.append(ra_arr,r['CP_RA'])
			dec_arr=np.append(dec_arr,r['CP_DEC'])
			id_arr=np.append(id_arr,r['REC_NO'])
			flux_arr=np.append(flux_arr,r['FULL_FLUX'])
			ztype=np.append(ztype,0)
			pdfs.append([np.array([r['SPEC_Z']]),np.array([1.])])
	temp = list(zip(ra_arr,dec_arr,z_arr,flux_arr,w_arr,id_arr,ztype))
	parsed = np.zeros((len(ra_arr),), dtype=[('ra', '<f8'),('dec', '<f8'),('z', '<f8'),('flux', '<f8'),('weight', '<f8'),('id', '<f8'),('ztype', '<i8')])
	parsed[:] = temp

	return parsed, np.array(pdfs)

