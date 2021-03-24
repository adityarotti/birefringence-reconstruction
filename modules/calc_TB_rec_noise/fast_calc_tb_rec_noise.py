import numpy as np
from tb_noise_sampL import rec_noise

def calc_tb_rec_noise(clte_thry,cltt_obs,clbb_obs,ellmin,ellmax,Lmax,nsamples=40):
    Larray=np.linspace(1,Lmax,nsamples,dtype=np.int).astype(np.float64)
    NLs=rec_noise(clte_thry[:ellmax+1],cltt_obs[:ellmax+1],clbb_obs[:ellmax+1],ellmin=2,ellmax=ellmax,lsize=nsamples,larray=Larray)
    NL=np.zeros(Lmax+1,dtype=np.float64)
    NL[1:]=np.interp(np.linspace(1,Lmax,Lmax),Larray,NLs)
    return NL
