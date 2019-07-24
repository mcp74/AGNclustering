import numpy as np
from Corrfunc.mocks import DDrppi_mocks,DDtheta_mocks

def w_theta(data,rand,bins):
    ra=data['ra']
    dec=data['dec']
    rar=rand['ra']
    decr=rand['dec']
    
    nbins = len(bins)-1

    DD_counts = DDtheta_mocks(autocorr=1, nthreads=2, binfile=bins, RA1=ra, DEC1=dec)
    DR_counts = DDtheta_mocks(autocorr=0, nthreads=2, binfile=bins, RA1=ra, DEC1=dec, RA2=rar, DEC2=decr)
    #RR_counts = DDtheta_mocks(autocorr=1, nthreads=2, binfile=bins, RA1=rar, DEC1=decr)

    N=len(data)
    N_rand=len(rand)
    
    cf = np.zeros(nbins)
    cf[:] = np.nan
    DR = DR_counts['npairs'] / (N*N_rand)
    DD = DD_counts['npairs'] / (N*(N-1))
    th = DD_counts['thetamax']
    nonzero = DR > 0
    cf[nonzero] = (DD[nonzero] / DR[nonzero] ) - 1.

    return th,cf