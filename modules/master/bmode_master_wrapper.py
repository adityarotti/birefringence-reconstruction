##################################################################################################
# Author: Aditya Rotti, Jodrell Bank Center for Astrophysics, University of Manchester
# Date created: 20 March 2020
# Date modified: 20 August 2020
##################################################################################################
import healpy as h
import numpy as np
import binned_master as bm
from astropy.io import fits
from scipy.interpolate import interp1d
import collections

class master_wrapper(object):
	def __init__(self,nside,mask,lmin,lmax,masklmax,dell,fwhm,bin_pad,step_bin=False,do_pwc=False):
		self.nside=nside
		self.mask=h.ud_grade(mask,self.nside)
		self.fsky=np.sum(mask)/np.size(mask)
		self.w1=np.sum(self.mask)*h.nside2pixarea(self.nside)/(4.*np.pi)
		self.w2=np.sum(self.mask**2.)*h.nside2pixarea(self.nside)/(4.*np.pi)
		self.w4=np.sum(self.mask**4.)*h.nside2pixarea(self.nside)/(4.*np.pi)
		self.lmin=lmin ; self.lmax=lmax
		self.masklmax=masklmax
		self.dell=dell
		self.fwhm=(fwhm/60.)*np.pi/180.
		self.bin_pad=bin_pad
		self.step_bin=step_bin
		self.do_pwc=do_pwc

		self.beam=h.gauss_beam(fwhm=self.fwhm,lmax=self.lmax)
		self.pwc=np.ones_like(self.beam)
		if self.do_pwc:
			self.pwc=h.pixwin(self.nside)[:self.lmax+1]

		self.mymstr=bm.binned_master(mask=self.mask,lmin=self.lmin,lmax=self.lmax,
								masklmax=self.masklmax,beam=self.beam*self.pwc,
								deltaell=self.dell,bin_pad=self.bin_pad,
								step_bin=self.step_bin)

	def setup_bbthry(self,ell,clbb_prim,clbb_lens):
		'''
		Pass tensor B-mode spectra corresponding to r=1.
		'''
		self.ell=ell
		self.clbb_prim=clbb_prim ; self.clbb_lens=clbb_lens
		fl=self.ell*(self.ell+1)/(2.*np.pi) ; fl[0]=1. ; fl[1]=1.
		self.clbb_prim=self.clbb_prim/fl
		self.clbb_lens=self.clbb_lens/fl
		self.lbin,self.bin_clbb_prim=self.mymstr.return_binned_spectra(self.clbb_prim)
		self.lbin,self.bin_clbb_lens=self.mymstr.return_binned_spectra(self.clbb_lens)
	
	def setup_eethry(self,ell,clee_lens):
		'''
		Pass tensor B-mode spectra corresponding to r=1.
		'''
		self.ell=ell
		self.clee_lens=clee_lens
		fl=self.ell*(self.ell+1)/(2.*np.pi) ; fl[0]=1. ; fl[1]=1.
		self.clee_lens=self.clee_lens/fl
		self.lbin,self.bin_clee_lens=self.mymstr.return_binned_spectra(self.clee_lens)

	def simulated_bmode_analysis(self,r=3.e-3,seed=0):
		np.random.seed(seed)
		bbsim=h.synfast(self.clbb_prim*r,nside=self.nside,fwhm=self.fwhm,pixwin=self.do_pwc,verbose=False)
		cl_bbsim=h.alm2cl(h.map2alm(bbsim*self.mask,lmax=2*self.nside))
		lbin,self.clbb_sim=self.mymstr.return_bmcs(cl_bbsim)

	def get_spectra_cov(self,filename,col_names=["obs","frg","noise","cmb"],nrlz=1000,compute_cov=False):
		stat=collections.OrderedDict()
		stat["ell"]=np.arange(self.lmax+1)
		stat["raw_spectra"]={}
		for icol,col in enumerate(col_names):
			temp_data=h.read_map(filename,icol,verbose=False,dtype=np.float64)
			stat["raw_spectra"][col]=h.alm2cl(h.map2alm(temp_data*self.mask,lmax=self.lmax))

		stat["lbin"]=self.lbin
		stat["spectra"]={}
		for col in col_names:
			temp,stat["spectra"][col]=self.mymstr.return_bmcs(stat["raw_spectra"][col])

		if compute_cov:
			stat["cov_est"]={}

			stat["cov_est"]["noise"]={}
#			clnse=stat["raw_spectra"]["noise"]/(self.fsky*(self.beam*self.pwc)**2.)
			clnse=self.mymstr.return_mcs(stat["raw_spectra"]["noise"])
			clmean,clcov=self.mymstr.return_binned_master_covariance(clnse,nrlz=nrlz)
			stat["cov_est"]["noise"]["clmean"]=clmean
			stat["cov_est"]["noise"]["cov"]=clcov
			
			stat["cov_est"]["frg"]={}
#			clfrg=stat["raw_spectra"]["frg"]/(self.fsky*(self.beam*self.pwc)**2.)
			clfrg=self.mymstr.return_mcs(stat["raw_spectra"]["frg"])
			clmean,clcov=self.mymstr.return_binned_master_covariance(clfrg,nrlz=nrlz)
			stat["cov_est"]["frg"]["clmean"]=clmean
			stat["cov_est"]["frg"]["cov"]=clcov

			stat["cov_est"]["cmb"]={}
			clmean,clcov=self.mymstr.return_binned_master_covariance(self.clbb_lens,nrlz=nrlz)
			stat["cov_est"]["cmb"]["clmean"]=clmean
			stat["cov_est"]["cmb"]["cov"]=clcov

		return stat
