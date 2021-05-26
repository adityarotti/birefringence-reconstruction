import numpy as np
import healpy as h
from matplotlib import pyplot as plt
from . import bmaster as ms

# Assumes that the spectra provided have l=0,1 included. Excludes those explicitly by starting from the second element of the array.
class binned_master(object):
	
	def __init__(self,mask,lmin,lmax,masklmax,beam=np.ones(4096,float),deltaell=20,bmin=[],bmax=[],step_bin=False,bin_pad=2):
		self.mask=mask
		self.nside=h.get_nside(self.mask)
		self.lmin=np.int(max(lmin,1))
		self.lmax=np.int(lmax)
		self.masklmax=masklmax
		self.beam=beam
		self.deltaell=deltaell
		self.totell=np.int(self.lmax-self.lmin+1)
		self.step_bin=step_bin
		self.bin_pad=bin_pad

		self.bmin=bmin
		if self.bmin!=[]:
			self.lmin=np.int(min(self.bmin))

		self.bmax=bmax
		if self.bmax!=[]:
			self.lmax=np.int(max(self.bmax))

		self.clmask=h.alm2cl(h.map2alm(self.mask,lmax=self.masklmax))
		self.mllp=ms.master.calc_kernel(self.clmask,self.lmin,self.lmax,self.masklmax)

		self.pbl=[]
		self.qlb=[]
		self.lbin=[]
		self.deltaell_bin=[]
		self.mbbp=[]

		#self.setup_binning_old(self.deltaell)
		self.setup_binning()
		if self.bin_pad<0:
			self.bin_pad=-self.nbin

	# Defining the projection and the deprojection operators
	def setup_binning_old(self,deltaell):
		totell=self.lmax-self.lmin+1
		nbin=np.int(totell/deltaell)
		self.pbl=np.zeros((nbin,totell),float)
		self.qlb=np.zeros((totell,nbin),float)
		self.qlb_nobeam=np.zeros((totell,nbin),float)
		self.lbin=[]
		self.deltaell_bin=[]
		self.lbin_low=[]
		self.lbin_high=[]

		for i in range(nbin):
			bmin=i*deltaell
			bmax=min(bmin+deltaell-1,self.lmax-self.lmin)
			ell=np.linspace(bmin+self.lmin,bmax+self.lmin,bmax-bmin+1)
			ellmin=min(ell) ; ellmax=max(ell)
			self.lbin_low=np.append(self.lbin_low,ellmin)
			self.lbin_high=np.append(self.lbin_high,ellmax)
			temp_bl=self.beam[int(ellmin):int(ellmax)+1]
			self.lbin=np.append(self.lbin,int(np.mean(ell)))
			norm=len(ell)
			if self.step_bin:
				f1=1./norm
				g1=1.
			else:
				f1=ell*(ell+1)/(2.*np.pi*norm)
				g1=2.*np.pi/(ell*(ell+1))
			self.pbl[i,bmin:bmax+1]=f1
			self.qlb[bmin:bmax+1,i]=g1*(temp_bl**2.)
			self.qlb_nobeam[bmin:bmax+1,i]=g1
			self.deltaell_bin=np.append(norm,self.deltaell_bin)
		
		self.mbbp=np.array(np.matrix(self.pbl)*np.matrix(self.mllp)*np.matrix(self.qlb))
		
	# Defining the projection and the deprojection operators
	def setup_binning(self):
		if self.bmin==[] and self.bmax==[]:
			self.nbin=np.int(self.totell*1./self.deltaell)
			while self.deltaell*self.nbin<self.lmax:
				self.nbin=self.nbin+1
			self.bmin=np.zeros(self.nbin,dtype="float")
			self.bmax=np.zeros(self.nbin,dtype="float")
			for i in range(self.nbin):
				self.bmin[i]=min(i*self.deltaell+self.lmin,self.lmax)
				self.bmax[i]=min(self.bmin[i]+self.deltaell-1,self.lmax)
		else:
			self.nbin=len(self.bmin)
			self.totell=np.int(self.lmax-self.lmin+1)

		self.pbl=np.zeros((self.nbin,self.totell),float)
		self.qlb=np.zeros((self.totell,self.nbin),float)
		self.qlb_nobeam=np.zeros((self.totell,self.nbin),float)
		self.lbin=[]
		self.deltaell_bin=[]

		for i in range(self.nbin):
			tbmin=np.int(self.bmin[i]) ; tbmax=np.int(self.bmax[i])
			ibmin=tbmin-np.int(self.lmin) ; ibmax=tbmax-np.int(self.lmin)
			ell=np.linspace(tbmin,tbmax,tbmax-tbmin+1)
			temp_bl=self.beam[tbmin:tbmax+1]
			self.lbin=np.append(self.lbin,(np.mean(ell)))
			norm=len(ell)
			if self.step_bin:
				f1=1./norm
				g1=1.
			else:
				f1=ell*(ell+1)/(2.*np.pi*norm) #(2.*ell+1)/(2.*np.pi*norm)ss
				g1=2.*np.pi/(ell*(ell+1)) #2.*np.pi/(2.*ell+1)
			self.pbl[i,ibmin:ibmax+1]=f1
			self.qlb[ibmin:ibmax+1,i]=g1*(temp_bl**2.)
			self.qlb_nobeam[ibmin:ibmax+1,i]=g1
			self.deltaell_bin=np.append(self.deltaell_bin,norm)

		self.mbbp=np.array(np.matrix(self.pbl)*np.matrix(self.mllp)*np.matrix(self.qlb))
		
	def return_mcs(self,cl):
		"""
		Returns the master corrected spectrum (mcs)
		"""
		mcl=ms.master.est_true_cl(cl[self.lmin:self.lmax+1],self.mllp,len(cl[self.lmin:self.lmax+1]))
		mcl=np.append(np.zeros(self.lmin,float),mcl)
		mcl=mcl/(self.beam[:self.lmax+1]**2.)
		return mcl

	def return_bmcs(self,cl):
		"""
		Returns the binned master corrected spectrum (bmcs)
		"""
		bcl=np.array(np.matrix(self.pbl)*np.transpose(np.matrix(cl[self.lmin:self.lmax+1])))[:,0]
		bcl=ms.master.est_true_cl(bcl,self.mbbp,len(bcl))

		#ubcl=np.array(np.matrix(self.qlb_nobeam)*np.transpose(np.matrix(bcl)))[:,0]
		#ubcl=np.append(np.zeros(self.lmin,float),ubcl)

		return self.lbin[:-self.bin_pad],bcl[:-self.bin_pad] #,ubcl

	def return_bmcs_sub_thry(self,cl,clthry=np.zeros(2048,float)):
		"""
		Returns the binned master corrected spectrum (bmcs)
		"""
		bcl=np.array(np.matrix(self.pbl)*np.transpose(np.matrix(cl[self.lmin:self.lmax+1])))[:,0]
		bcl=ms.master.est_true_cl(bcl,self.mbbp,len(bcl))
		bclthry=np.array(np.matrix(self.pbl)*np.transpose(np.matrix(clthry[self.lmin:self.lmax+1])))[:,0]
		bcl=bcl-bclthry

		#ubcl=np.array(np.matrix(self.qlb_nobeam)*np.transpose(np.matrix(bcl)))[:,0]
		#ubcl=np.append(np.zeros(self.lmin,float),ubcl)

		return self.lbin,bcl #,ubcl
	
	def return_binned_master_covariance(self,clthry,nrlz=1000):
		clthry=clthry[:self.lmax+1]*(self.beam[:self.lmax+1]**2.)
		cl_raw=np.zeros((nrlz,len(self.lbin)-self.bin_pad),np.float64)
		for i in range(nrlz):
			d=h.synfast(clthry,self.nside,verbose=False)
			cld=h.alm2cl(h.map2alm(d*self.mask,lmax=self.lmax))
			lbin,cl_raw[i,:]=self.return_bmcs(cld)
			
		clmean=np.mean(cl_raw,axis=0)
		cov=np.zeros((len(self.lbin)-self.bin_pad,len(self.lbin)-self.bin_pad),np.float64)
		for i in range(nrlz):
			cov=cov + np.matmul(np.matrix(cl_raw[i,:]-clmean).T,np.matrix(cl_raw[i,:]-clmean))
		cov=cov/(nrlz-1)
		return clmean,cov
	
	def return_binned_spectra(self,cl):
		bcl=np.array(np.matrix(self.pbl)*np.transpose(np.matrix(cl[self.lmin:self.lmax+1])))[:,0]
		return self.lbin[:-self.bin_pad],bcl[:-self.bin_pad]

	def plot_mask(self,maskindex,pathout):
		h.orthview(self.mask,title="",rot=(0,90))
		filename = pathout + "disc_mask" + "_mi" + str(maskindex) + ".pdf"
		plt.savefig(filename,dpi=150,bbox_inches='tight')
		plt.close()




	
		

