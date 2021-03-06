import numpy as np
from .eb_noise_sampL import rec_noise


def calc_eb_rec_noise(clee_thry,clee_obs,clbb_obs,ellmin,ellmax,Lmax,nsamples=40):
    Larray=np.linspace(1,Lmax,nsamples,dtype=np.int).astype(np.float64)
    NLs=rec_noise(clee_thry[:ellmax+1],clee_obs[:ellmax+1],clbb_obs[:ellmax+1],ellmin=ellmin,ellmax=ellmax,lsize=nsamples,larray=Larray)
    NL=np.zeros(Lmax+1,dtype=np.float64)
    NL[1:]=np.interp(np.linspace(1,Lmax,Lmax),Larray,NLs)
    return NL
