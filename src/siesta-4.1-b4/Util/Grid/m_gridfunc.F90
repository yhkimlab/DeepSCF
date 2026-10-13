! ---
! Copyright (C) 1996-2016	The SIESTA group
!  This file is distributed under the terms of the
!  GNU General Public License: see COPYING in the top directory
!  or http://www.gnu.org/copyleft/gpl.txt .
! See Docs/Contributors.txt for a list of contributors.
! ---
!
module m_gridfunc

  implicit          none

  integer, parameter, private :: dp = selected_real_kind(10,100)
  integer, parameter, private :: sp = selected_real_kind(5,10)
  integer, parameter          :: grid_p = sp  ! NOTE this

  type, public :: gridfunc_t
     real(dp)              ::  cell(3,3)   ! by columns, in bohr
     integer               ::  n(3)
     integer               ::  nspin
     real(grid_p), allocatable ::  val(:,:,:,:)
  end type gridfunc_t

      public :: write_gridfunc, read_gridfunc, clean_gridfunc
#ifdef CDF
      public :: read_gridfunc_netcdf, write_gridfunc_netcdf
#endif
         
      private

 CONTAINS

   subroutine clean_gridfunc(gf)
     type(gridfunc_t), intent(inout) :: gf

     if (allocated(gf%val)) then
        deallocate(gf%val)
     endif
     gf%n(:) = 0
     gf%cell(:,:) = 0.0_dp

   end subroutine clean_gridfunc

   subroutine write_gridfunc(gf,fname)
     type(gridfunc_t), intent(in) :: gf
     character(len=*), intent(in) :: fname

     integer :: n(3)
     integer :: isp, ix, iy, iz
     integer :: iu

     n = gf%n

     call get_lun(iu)
     open(unit=iu,file=fname,form="unformatted",status="unknown", &
          position="rewind",action="write")
     write(iu) gf%cell
     write(iu) n, gf%nspin
     do isp=1,gf%nspin
        do iz=1,n(3)
           do iy=1,n(2)
              write(iu) (gf%val(ix,iy,iz,isp),ix=1,n(1))
           enddo
        enddo
     enddo
     close( iu )

   end subroutine write_gridfunc

   subroutine read_gridfunc(fname,gf)
     type(gridfunc_t), intent(inout) :: gf
     character(len=*), intent(in) :: fname

     integer :: n(3) 
     integer :: isp, ix, iy, iz
     integer :: iu

     call clean_gridfunc(gf)

     call get_lun(iu)
     open(unit=iu,file=fname,form="unformatted",status="old", &
          position="rewind",action="read")
     read(iu) gf%cell
     read(iu) gf%n, gf%nspin
     n = gf%n
     allocate(gf%val(n(1),n(2),n(3),gf%nspin))

     do isp=1,gf%nspin
        do iz=1,n(3)
           do iy=1,n(2)
              read(iu) (gf%val(ix,iy,iz,isp),ix=1,n(1))
           enddo
        enddo
     enddo
     close( iu )

   end subroutine read_gridfunc

#ifdef CDF

subroutine write_gridfunc_netcdf(gf,filename,description)
use netcdf

implicit none

character(len=*), intent(in) :: filename
type(gridfunc_t), intent(in)    :: gf
character(len=*), intent(in), optional :: description

integer  :: ncid 
integer  :: xyz_id, step_id, abc_id, spin_id
integer  :: cell_id, n1_id, n2_id, n3_id, gridfunc_id

integer   ::    nspin  ! Number of spins 
integer   ::    n(3)
integer   ::   ispin, iostat, ix, iy, iz, iret

!-----------------------------------------------------

call check( nf90_create(filename,NF90_CLOBBER,ncid))
       iret = nf90_def_dim(ncid,'xyz',3,xyz_id)
       call check(iret)
       iret = nf90_def_dim(ncid,'abc',3,abc_id)
       call check(iret)
       iret = nf90_def_dim(ncid,'spin',gf%nspin,spin_id)
       call check(iret)

       n(:) = gf%n(:)

       iret = nf90_def_dim(ncid,'n1',n(1),n1_id)
       call check(iret)
       iret = nf90_def_dim(ncid,'n2',n(2),n2_id)
       call check(iret)
       iret = nf90_def_dim(ncid,'n3',n(3),n3_id)
       call check(iret)

       iret = nf90_def_var(ncid,'cell',nf90_float,(/xyz_id,abc_id/),cell_id)
       call check(iret)
       iret = nf90_put_att(ncid,cell_id,'Description', &
               "Cell vectors in Bohr: xyz, abc")
       call check(iret)

       iret = nf90_def_var(ncid,'gridfunc',nf90_float,(/n1_id,n2_id,n3_id,spin_id/),gridfunc_id)
       call check(iret)
       if (present(description)) then
          iret = nf90_put_att(ncid,gridfunc_id,'Description', &
               trim(description))
       else
          iret = nf90_put_att(ncid,gridfunc_id,'Description', &
               "Grid function -- ")
       endif
       call check(iret)

       iret = nf90_enddef(ncid)
       call check(iret)
!
       iret = nf90_put_var(ncid, cell_id, gf%cell, start = (/1, 1 /), count = (/3, 3/) )
       call check(iret)

      iret = nf90_put_var(ncid, gridfunc_id, gf%val, start = (/1, 1, 1, 1 /), &
           count = (/n(1), n(2), n(3), nspin/) )
      call check(iret)
   
     call check( nf90_close(ncid) )

end subroutine write_gridfunc_netcdf
!--------

subroutine read_gridfunc_netcdf(filename,gf)
use netcdf

implicit none

character(len=*), intent(in) :: filename
type(gridfunc_t), intent(inout)    :: gf

integer  :: ncid 
integer  :: xyz_id, step_id, abc_id, spin_id
integer  :: cell_id, n1_id, n2_id, n3_id, gridfunc_id

integer   ::    nspin  ! Number of spins 
integer   ::    n(3)
integer   ::   ispin, iostat, ix, iy, iz, iret

real(dp)  ::    cell(3,3)
!-----------------------------------------------------

call clean_gridfunc(gf)

call check( nf90_open(filename,NF90_NOWRITE,ncid))
       call check( nf90_inq_dimid(ncid,'spin',spin_id) )
       call check( nf90_inquire_dimension(ncid, dimid=spin_id, len=nspin) )

       call check( nf90_inq_dimid(ncid,'n1',n1_id) )
       call check( nf90_inquire_dimension(ncid, dimid=n1_id, len=n(1)) )
       call check( nf90_inq_dimid(ncid,'n2',n2_id) )
       call check( nf90_inquire_dimension(ncid, dimid=n2_id, len=n(2)) )
       call check( nf90_inq_dimid(ncid,'n3',n3_id) )
       call check( nf90_inquire_dimension(ncid, dimid=n3_id, len=n(3)) )

       call check( nf90_inq_varid(ncid, "cell", cell_id) )
       call check( nf90_inq_varid(ncid, "gridfunc", gridfunc_id) )

       iret = nf90_get_var(ncid, cell_id, cell, start = (/1, 1 /), &
                        count = (/3, 3/) )
       call check(iret)

   gf%n(:) = n(:)
   gf%cell = cell

   allocate(gf%val(n(1),n(2),n(3),nspin))

   iret = nf90_get_var(ncid, gridfunc_id, gf%val, start = (/1, 1, 1, 1 /), &
        count = (/n(1), n(2), n(3), nspin/) )
   call check(iret)
   
   call check( nf90_close(ncid) )

end subroutine read_gridfunc_netcdf

subroutine check(code)
use netcdf, only: nf90_noerr, nf90_strerror
integer, intent(in) :: code
if (code /= nf90_noerr) then
  print *, "netCDF error: " // NF90_STRERROR(code)
  STOP
endif
end subroutine check

#endif /* CDF */

 subroutine get_lun(lun)
   integer, intent(out) :: lun

   logical :: busy
   do lun = 1, 99
      inquire(unit=lun,opened=busy)
      if (.not. busy) RETURN
   enddo
   call die("Cannot get free unit")
 end subroutine get_lun

 end module m_gridfunc

