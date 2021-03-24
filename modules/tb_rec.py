import numpy as np
import healpy as h
import collections
from calc_TB_rec_noise.fast_calc_tb_rec_noise import calc_tb_rec_noise as rec_noise

class opt_tb_qe(object):
    def __init__(self,obs,clthry,ellmax,Lmax,Lsampling=40):
		self.ellmax=ellmax
		self.Lmax=Lmax
		self.cltt=clthry[0][:self.ellmax+1]
		self.clbb=clthry[2][:self.ellmax+1]
		self.clte=clthry[3][:self.ellmax+1]
		self.Lsampling=Lsampling

		self.N_L_ideal=rec_noise(self.clte,self.cltt,self.clbb,ellmin=2,ellmax=self.ellmax,Lmax=self.Lmax)
		self.myobs=np.copy(obs)
		self.nside=h.get_nside(self.myobs)
	
    def tb_reconstruct(self):
		self.teb_alm=h.map2alm(self.myobs,pol=True,lmax=self.ellmax)
		zrs=np.zeros(len(self.teb_alm[0]),dtype=np.float64)
		self.clteb=h.alm2cl(self.teb_alm)
		self.N_L=rec_noise(self.clte,self.clteb[0],self.clteb[2],ellmin=2,ellmax=self.ellmax,Lmax=self.Lmax)

		T_filter=self.clte[2:]/self.clteb[0][2:] ; T_filter=np.append([0.,0.],T_filter)
		fil_Talm=h.almxfl(np.conj(self.teb_alm[0]),fl=T_filter,inplace=False)
		qt,ut=h.alm2map_spin([fil_Talm,zrs-1.j*zrs],self.nside,2,lmax=self.ellmax)

		B_filter=1./(self.clteb[2][2:]) ; B_filter=np.append([0.,0.],B_filter)
		fil_Balm=h.almxfl(np.conj(self.teb_alm[2]),fl=B_filter,inplace=False)
		qb,ub=h.alm2map_spin([fil_Balm,zrs-1.j*zrs],self.nside,2,lmax=self.ellmax)
		
		rec_alpha_alm=h.map2alm(qt*qb + ut*ub,lmax=self.Lmax)
		rec_alpha_alm=h.almxfl(np.conj(rec_alpha_alm),fl=self.N_L,inplace=False)

		self.Cl_rec_alpha=h.alm2cl(rec_alpha_alm)
		self.rec_alpha=h.alm2map(rec_alpha_alm,nside=self.nside,lmax=self.Lmax,verbose=False)

		self.wf=self.Cl_rec_alpha-self.N_L ; self.wf[self.wf<0.]=0. ; self.wf[1:]=self.wf[1:]/self.Cl_rec_alpha[1:] ; self.wf[0]=0.
		self.wf_rec_alpha=h.alm2map(h.almxfl(rec_alpha_alm,fl=self.wf,inplace=False),nside=self.nside,lmax=self.Lmax,verbose=False)


