import numpy as np
import healpy as h
import collections
from calc_EB_rec_noise.fast_calc_eb_rec_noise import calc_eb_rec_noise as rec_noise

class opt_eb_qe(object):
    def __init__(self,obs,clthry,ellmax,Lmax,Lsampling=40):
		self.ellmax=ellmax
		self.Lmax=Lmax
		self.clee=clthry[1][:self.ellmax+1]
		self.clbb=clthry[2][:self.ellmax+1]
		self.Lsampling=Lsampling

		self.N_L_ideal=rec_noise(self.clee,self.clee,self.clbb,ellmin=2,ellmax=self.ellmax,Lmax=self.Lmax)
		self.myobs=np.copy(obs)
		self.nside=h.get_nside(self.myobs)
	
    def eb_reconstruct(self):
		self.eb_alm=h.map2alm_spin(self.myobs[1:3],2,lmax=self.ellmax)
		zrs=np.zeros(len(self.eb_alm[0]),dtype=np.float64)
		self.cleb=h.alm2cl(self.eb_alm)
		self.N_L=rec_noise(self.clee,self.cleb[0],self.cleb[1],ellmin=2,ellmax=self.ellmax,Lmax=self.Lmax)

		E_filter=self.clee[2:]/self.cleb[0][2:] ; E_filter=np.append([0.,0.],E_filter)
		fil_Ealm=h.almxfl(np.conj(self.eb_alm[0]),fl=E_filter,inplace=False)
		qe,ue=h.alm2map_spin([fil_Ealm,zrs-1.j*zrs],self.nside,2,lmax=self.ellmax)

		B_filter=1./(self.cleb[1][2:]) ; B_filter=np.append([0.,0.],B_filter)
		fil_Balm=h.almxfl(np.conj(self.eb_alm[1]),fl=B_filter,inplace=False)
		qb,ub=h.alm2map_spin([fil_Balm,zrs-1.j*zrs],self.nside,2,lmax=self.ellmax)
		
		rec_alpha_alm=h.map2alm(qe*qb + ue*ub,lmax=self.Lmax)
		rec_alpha_alm=h.almxfl(np.conj(rec_alpha_alm),fl=self.N_L,inplace=False)

		self.Cl_rec_alpha=h.alm2cl(rec_alpha_alm)
		self.rec_alpha=h.alm2map(rec_alpha_alm,nside=self.nside,lmax=self.Lmax,verbose=False)

		self.wf=self.Cl_rec_alpha-self.N_L ; self.wf[self.wf<0.]=0. ; self.wf[1:]=self.wf[1:]/self.Cl_rec_alpha[1:] ; self.wf[0]=0.
		self.wf_rec_alpha=h.alm2map(h.almxfl(rec_alpha_alm,fl=self.wf,inplace=False),nside=self.nside,lmax=self.Lmax,verbose=False)

