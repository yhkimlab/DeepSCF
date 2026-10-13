! ---
! Copyright (C) 1996-2016	The SIESTA group
!  This file is distributed under the terms of the
!  GNU General Public License: see COPYING in the top directory
!  or http://www.gnu.org/copyleft/gpl.txt .
! See Docs/Contributors.txt for a list of contributors.
! ---
module version_info

implicit none

integer, dimension(3), save  :: num_version = (/0,0,0/)
character(len=*), parameter :: version_str =  &
"v4.1-b4"
character(len=*), parameter :: siesta_arch= &
"x86_64-unknown-linux-gnu--unknown"
character(len=*), parameter :: fflags= &
"mpiifort -g -O2 -fPIC -xSSE4.2 -fp-model source"
character(len=*), parameter :: fppflags= &
"-DFC_HAVE_FLUSH -DFC_HAVE_ABORT -DMPI  -DCDF -DNCDF -DNCDF_4 -DTBTRANS"
character(len=*), parameter :: libs= &
"libncdf.a libfdict.a -Wl,--start-group /opt/intel/oneapi/mkl/2022.1.0/lib/intel64/libmkl_scalapack_lp64.a /opt/intel/oneapi/mkl/2022.1.0/lib/intel64/libmkl_intel_lp64.a /opt/intel/oneapi/mkl/2022.1.0/lib/intel64/libmkl_core.a /opt/intel/oneapi/mkl/2022.1.0/lib/intel64/libmkl_sequential.a /opt/intel/oneapi/mkl/2022.1.0/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group -lpthread   /opt/intel/oneapi/mkl/2022.1.0/lib/intel64/libmkl_blas95_lp64.a -Wl,--start-group /opt/intel/oneapi/mkl/2022.1.0/lib/intel64/libmkl_intel_lp64.a /opt/intel/oneapi/mkl/2022.1.0/lib/intel64/libmkl_sequential.a /opt/intel/oneapi/mkl/2022.1.0/lib/intel64/libmkl_core.a -Wl,--end-group -L/home2/rong/opt/netcdf_build/build/build/netcdf/4.4.1.1/lib -L/home2/rong/opt/netcdf_build/build/build/netcdf/4.4.1.1/lib -L/home2/rong/opt/netcdf_build/build/build/hdf5/1.8.19/lib -L/home2/rong/opt/netcdf_build/build/build//zlib/1.2.11/lib /home2/rong/opt/netcdf_build/build/build/netcdf/4.4.1.1/lib/libnetcdff.a /home2/rong/opt/netcdf_build/build/build/netcdf/4.4.1.1/lib/libnetcdf.a /home2/rong/opt/netcdf_build/build/build/hdf5/1.8.19/lib/libhdf5_hl.a /home2/rong/opt/netcdf_build/build/build/hdf5/1.8.19/lib/libhdf5.a /home2/rong/opt/netcdf_build/build/build//zlib/1.2.11/lib/libz.a -lcurl -lnetcdff -lnetcdf -lhdf5_hl -lhdf5 -lz"

private
public :: num_version, version_str
public :: siesta_arch, fflags, fppflags, libs

end module version_info
!================================================================

subroutine prversion

use version_info
implicit none

#ifdef TBT_PHONON
write(6,'(2a)') "PHtrans Version: ", trim(version_str)
#else
write(6,'(2a)') "TBtrans Version: ", trim(version_str)
#endif
write(6,'(2a)') 'Architecture  : ', trim(siesta_arch)
write(6,'(2a)') 'Compiler flags: ', trim(fflags)
write(6,'(2a)') 'PP flags      : ', trim(fppflags)
write(6,'(2a)') 'Libraries     : ', trim(libs)

#ifdef MPI
write(6,'(a)') 'PARALLEL version'
#else
write(6,'(a)') 'SERIAL version'
#endif

!$OMP parallel
!$OMP master
!$write(*,'(a)') 'THREADED version'
#ifdef _OPENMP
!$write(*,'(a,i0)') '* OpenMP version ', _OPENMP
#endif
!$OMP end master
!$OMP end parallel

#ifdef USE_GEMM3M
write(6,'(a)') 'GEMM3M support'
#endif
#ifdef CDF
write(6,'(a)') 'NetCDF support'
#endif
#ifdef NCDF_4
write(6,'(a)') 'NetCDF-4 support'
#ifdef NCDF_PARALLEL
write(6,'(a)') 'NetCDF-4 MPI-IO support'
#endif
#endif
#if defined(ON_DOMAIN_DECOMP) || defined(SIESTA__METIS)
write(6,'(a)') 'METIS ordering support'
#endif

end subroutine prversion
!----------------------------------------------------------

subroutine get_version(v)
  use version_info
  implicit none
  integer, intent(out)  :: v(3)
  v = num_version
end subroutine get_version

