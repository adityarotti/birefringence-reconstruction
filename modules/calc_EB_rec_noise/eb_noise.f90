##################################################################################################
# Author: Aditya Rotti, Jodrell Bank Center for Astrophysics, University of Manchester
# Date created: 24 March 2021
# Date modified: 24 March 2021
##################################################################################################

module rec_noise

contains

!########################################################################
subroutine calc_eb_noise(clee_thry,clee_obs,clbb_obs,ellmin,ellmax,Lmax,rec_noise)
implicit none

integer*4 :: i, j, k, ier
integer*4, intent(in) :: ellmin,ellmax,Lmax
integer*8, parameter:: ndim=5000
real*8 :: L,lp, l1, k1min, k1max,tempvar, wig3j(ndim)
real*8, intent(in), dimension(0:ellmax) :: clee_thry,clee_obs,clbb_obs
real*8, intent(out) :: rec_noise(0:Lmax)
real*8, parameter :: pi=3.141592653589793d0

rec_noise(:)=0.d0

do i=1,Lmax
   L=float(i)
   do j=1,ellmax-ellmin+1
      lp=float(ellmin+j-1)
      call drc3jj(L,lp,0.d0,2.d0,k1min,k1max,wig3j,ndim,ier)
      do k=1,int(min(k1max,ellmax*1.d0)-k1min)+1
         l1=k1min+float(k-1)
         tempvar=(clee_thry(int(lp))**2.)/(clbb_obs(int(l1))*clee_obs(int(lp)))
         tempvar=tempvar*(wig3j(k)**2.d0)*(2.d0*lp+1.d0)*(2.d0*l1+1.d0)*4.d0/(4.d0*pi)
         if (mod(int(L+lp+l1),2).eq.0) then
!         	write(*,*) L,lp,l1,"Even",wig3j(k)
         	rec_noise(int(L))=rec_noise(int(L)) + tempvar
         end if
      enddo
   enddo
   rec_noise(int(L))=1.d0/rec_noise(int(L))
enddo

!rec_noise=1./rec_noise
!rec_noise(0)=0.d0 ; rec_noise(1)=0.d0

end subroutine calc_eb_noise
!########################################################################

end module rec_noise
