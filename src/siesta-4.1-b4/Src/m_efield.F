! 
! Copyright (C) 1996-2016	The SIESTA group
!  This file is distributed under the terms of the
!  GNU General Public License: see COPYING in the top directory
!  or http://www.gnu.org/copyleft/gpl.txt.
! See Docs/Contributors.txt for a list of contributors.
!
      module m_efield

!     This module implements routines to deal with an external
!     electric field.
!
!     The field required can be specified in two possible ways:
!
!     1. Through and FDF block 'ExternalElectricField'. For example:
!
!       %block ExternalElectricField
!           0.000  0.000  3.000  V/Ang
!       %endblock ExternalElectricField

!    2. Through the FDF Option 'SlabDipoleCorrection'. 
!    If 'true', the program will then calculate at every SCF step
!    the electric field required to compensate the dipole of the
!    system. The potential added to the grid corresponds to that
!    of a dipole layer in the middle of the vacuum layer. For slabs,
!    this exactly compensates the electric field at the vacuum created
!    by the dipole moment of the system, thus allowing to deal with
!    asymmetric slabs (and compute properties such as the work funcion
!    of each of the surfaces).
!    See L. Bengtsson, PRB 59, 12301 (1999), DOI: 10.1103/PhysRevB.59.12301

      use precision, only: dp
      use sys,       only: die

      implicit none
      logical, public  :: acting_efield = .false.
      logical, public  :: dipole_correction = .false.
      real(dp), public :: user_specified_field(3) =
     &     (/ 0.0_dp, 0.0_dp, 0.0_dp /)

      public :: initialize_efield
      public :: get_field_from_dipole, add_potential_from_field

      private

      CONTAINS

      subroutine get_user_specified_field(input_field)

      use fdf, only: fdf_convfac, fdf_block, fdf_bline, block_fdf
      use fdf, only: parsed_line, fdf_bvalues, fdf_bnames, fdf_bmatch

      real(dp),   intent(out)     :: input_field(3)

      type(block_fdf)            :: bfdf
      type(parsed_line), pointer :: pline => null()

      character(len=132)         :: eunits

      real(dp)                    :: cfactor
      integer                     :: ix

      input_field(1:3) = 0.0_dp

      if (.not. fdf_block('ExternalElectricField',bfdf) ) RETURN
      do while ( fdf_bline(bfdf,pline) )
        if (.not. fdf_bmatch(pline,"vvvn"))
     $     call die("Wrong format in ElectricField block")
        eunits = fdf_bnames(pline,1)
        cfactor = fdf_convfac(eunits,'Ry/Bohr/e')
        do ix = 1,3
           input_field(ix) = fdf_bvalues(pline,ix) * cfactor
        enddo
      enddo

      end subroutine get_user_specified_field
!----------------------------------------------------------------
       
      subroutine orthogonalize_efield(input_field,orthog_field,orthog,
     &     nbcell)

!     Orthogonalizes the field to the bulk directions

      use siesta_geom, only : ucell, xa, na_u, isa

      real(dp), intent(in)  :: input_field(3)
      real(dp), intent(out) :: orthog_field(3)
      logical,  intent(out) :: orthog ! originally orthogonal?
      integer,  intent(out) :: nbcell ! shape of cell


!     tolerance for bulk components of the electric field
      real(dp), parameter    :: tol = 1.0d-12

      real(dp)  :: eb1, eb2, eb3, b1xb2(3), bcell(3,3)
      integer   :: ix
      character(len=8) :: shape

      real(dp), external :: ddot

      call shaper( ucell, na_u, isa, xa, shape, nbcell, bcell )
      orthog = .true.
      if (nbcell .eq. 1) then
         eb1 = ddot(3,input_field,1,bcell,1) / ddot(3,bcell,1,bcell,1)
         if (abs(eb1) .gt. tol) then
            orthog = .false.
            do ix = 1,3
               orthog_field(ix) = input_field(ix) - eb1 * bcell(ix,1)
            enddo
         endif
      elseif (nbcell .eq. 2) then
         eb1 = ddot(3,input_field,1,bcell(1,1),1)/
     .        ddot(3,bcell(1,1),1,bcell(1,1),1)
         eb2 = ddot(3,input_field,1,bcell(1,2),1)/
     .        ddot(3,bcell(1,2),1,bcell(1,2),1)
         if (abs(eb1).gt.tol .or. abs(eb2).gt.tol) then
            orthog = .false.
            call cross( bcell(1,1), bcell(1,2), b1xb2 )
            eb3 = ddot(3,input_field,1,b1xb2,1)/ddot(3,b1xb2,1,b1xb2,1)
            do ix = 1,3
               orthog_field(ix) = eb3 * b1xb2(ix)
            enddo
         endif
      elseif (nbcell .eq. 3) then
         orthog = .false.
         do ix = 1,3
            orthog_field(ix) = 0.0_dp
         enddo
      endif
      end subroutine orthogonalize_efield
!----------------------------------------------------------------

      subroutine initialize_efield()
!
!     Initializations that can be done once and for all
!     It sets the module variables:
!          user_specified_field
!          acting_efield
!          dipole_correction
!

      use parallel,     only: ionode
      use siesta_cml,   only: cml_p, cmlAddProperty, mainXML
      use fdf,          only: fdf_get, fdf_bnames, fdf_block
      use fdf,          only: block_fdf, parsed_line, fdf_bline
      use fdf,          only: fdf_bmatch, fdf_convfac, fdf_bvalues
      use units,        only: Ang, eV
      use m_cite, only: add_citation
      use m_ts_global_vars,only: TSmode

      real(dp) :: input_field(3), orthog_field(3), cfactor
      logical  :: orthog
      integer :: ix
      integer :: nbcell
      character(len=20) :: eunits
      type(block_fdf)            :: bfdf
      type(parsed_line), pointer :: pline => null()

      real(dp), external :: ddot

      call get_user_specified_field(input_field)
      acting_efield = (ddot(3,input_field,1,input_field,1) /= 0.0_dp)

      ! always orthogonalize retrieve the shape of the cell
      call orthogonalize_efield(input_field,orthog_field,orthog,
     &     nbcell)
      
      if (acting_efield) then

         if (orthog) then
            if (ionode) then
               write(6,'(/,a,3f12.6,a)')
     $              'efield: External electric field =',
     $              input_field/eV*Ang, ' eV/Ang/e'
            endif
            user_specified_field(1:3) = input_field(1:3)
         else
            if (ionode) then
               write(6,'(a,(/,a,3f12.6))')
     .         'efield: WARNING: Non zero bulk electric field.',
     .         'efield: Input field (eV/Ang/e) =', input_field/eV*Ang,
     .         'efield: Orthogonalized field   =', orthog_field/eV*Ang
            endif
            user_specified_field(1:3) = orthog_field(1:3)
         endif
         if (cml_p) call cmlAddProperty(xf=mainXML,
     $           value=user_specified_field, 
     .           dictref='siesta:elfield', 
     .           units='siestaUnits:Ry_Bohr_e')
      endif

      acting_efield =
     $ (ddot(3,user_specified_field,1,user_specified_field,1) /= 0.0_dp)

      ! Only allow slab dipole corrections for slabs
      ! We also allow dipole corrections for transiesta calculations
      ! with a well-defined transport direction.
      if ( nbcell == 3 ) then
         ! Any fully periodic system should not allow dipole corrections
         dipole_correction = .false.
      else if ( nbcell == 2 .or. TSmode ) then
         ! Note that we default to enable the correction if there is
         ! an external field, or a transiesta calculation along the
         ! semi-infinite lead-direction for 1-probe calculations.
         ! For 1-electrode calculations it may make sense to apply
         ! this correction.
         dipole_correction = fdf_get("SlabDipoleCorrection",
     &        acting_efield)
         if (acting_efield .and. (.not. dipole_correction)) then
            if (ionode) write(6,'(/,(a))')
     .        'efield: WARNING!',
     .        'efield: SlabDipoleCorrection is .false. with an external'
     .           //' efield.',
     .        'efield: For correct physics SlabDipoleCorrection should'
     .           //' be .true.',
     .        'efield: This is only for backwards compatibility!'
         end if
      else
         dipole_correction = .false.
      end if

      if ( dipole_correction ) then
         if ( IONode ) then
            call add_citation("10.1103/PhysRevB.59.12301")
         end if
         if (ionode) write(6,'(/,(a))')
     .        'efield: SlabDipoleCorrection enabled.',
     .        'efield: A dipole layer will be introduced in the vacuum',
     .        'efield: region to compensate the system dipole'
         acting_efield = .true.
      end if

      end subroutine initialize_efield
!----------------------------------------------------------------

      function get_field_from_dipole(dipole, cell) result(efield)

      use units,       only: pi

      real(dp), intent(in)  :: dipole(3)
      real(dp), intent(in)  :: cell(3,3)
      real(dp)              :: efield(3)

      real(dp), external  :: volcel

      efield(1:3) = -4.0_dp * pi * dipole(1:3) * 2.0_dp / volcel(cell)

      end function get_field_from_dipole
!----------------------------------------------------------------

      subroutine add_potential_from_field(efield, cell, na, isa, xa,
     $     meshG, nsm, V)

c     Adds the potential created by an external electric field.
c     Written by J.M.Soler. Feb. 1998.
c     Modified to operate only on the sub-matrix of the potential stored
c     locally. J.D.Gale March 1999.
c     Modularized by A. Garcia, Nov. 2009
c     ********* Input ******************************************************
c     real*8  cell(3,3) : Unit cell vectors
c     integer na        : Number of atoms
c     integer isa(na)   : Atomic species indexes
c     real*8  xa(3,na)  : Atomic positions (cartesian coordinates)
c     integer meshG(3)  : Number of mesh divisions in each cell direction
c     integer nsm       : Number of sub-mesh points along each axis
c     real*8  field(3)  : Electric field
c     ********* Input and output *******************************************
c     real    v(*)      : Electron potential, to which that created by the
c     electric field is added. Notice single precision.
c     ********* Units ******************************************************
c     Distances in Bohr
c     Energies in Rydbergs
c     Electric field in Ry/Bohr
c     Dipoles in electrons*Bohr
c     ********* Behaviour **************************************************
c     The sign of the potential is that for electrons (v=+E*x), i.e. 
c     opposite to that of the conventional electrostatic potential.
c     Notice that the potential is not initialized.
c     Bulk electric fields are not allowed. If the specified electric field
c     is not orthogonal to all bulk directions, it is orthogonalized, and
c     a warning message is printed.
c     The electric field produces a discontinuity of the potential in the
c     periodic cell, which is automatically placed in the middle of the
c     vacuum region.

      use precision, only: dp, grid_p

      use atmfuncs,     only : rcut
      use mesh,         only : meshLim

      implicit          none

      integer, intent(in)          ::  na, isa(na), meshG(3), nsm
      real(dp), intent(in)         ::  cell(3,3), efield(3), xa(3,na)
      real(grid_p), intent(inout)  ::  v(*)

      integer           i0(3), i1, i2, i3, ia, imesh,
     $     is, ix,
     .     j1, j2, j3, meshl(3),
     .     Xoffset, Yoffset, Zoffset, i30, i20, i10

      real(dp)          dplane(3),
     .     f(3), rc, rcell(3,3), v0,
     .     xfrac, xmax(3), xmean, xmin(3)

      real(dp), external :: ddot
      external          reclat


C     Find the origin of a shifted cell, with the system centered in it
C     This is done at every call, because of possible atomic movements

C     Find reciprocal unit cell and distance between lattice planes
      call reclat( cell, rcell, 0 )
      do ix = 1,3
         dplane(ix) = 1.0_dp /
     $        sqrt(ddot(3,rcell(1,ix),1,rcell(1,ix),1))
      enddo

C     Find the geometric center of the system
      do ix = 1,3
         xmin(ix) =  1.0e30_dp
         xmax(ix) = -1.0e30_dp
      enddo
      do ia = 1,na
         is = isa(ia)
         rc = rcut(is,0)
         do ix = 1,3
            xfrac = ddot(3,xa(1,ia),1,rcell(1,ix),1)
            xmin(ix) = min( xmin(ix), xfrac-rc/dplane(ix) )
            xmax(ix) = max( xmax(ix), xfrac+rc/dplane(ix) )
         enddo
      enddo

C     Find the mesh index of the origin of the shifted cell
      do ix = 1,3
         xmean = (xmin(ix) + xmax(ix)) / 2
         i0(ix) = nint( (xmean-0.5_dp) * meshG(ix) )
      end do

C     Ensure that all indices become positive in the below loop
      do ix = 1 , 3
         i0(ix) = i0(ix) - 10 * meshG(ix)
      end do      

C     Find the electric field in mesh coordinates, so that
C     v = efield*x = f*index
      do ix = 1,3
         f(ix) = ddot(3,efield,1,cell(1,ix),1) / max( meshG(ix), 1 )
      enddo

C     Find the potential at the origin of the shifted cell, so that
C     the potential is zero at the center of the cell

      v0 = - 0.5_dp*(f(1)*meshG(1) + f(2)*meshG(2) + f(3)*meshG(3))

      meshl(1) = (meshLim(2,1) - meshLim(1,1)+1)*nsm
      meshl(2) = (meshLim(2,2) - meshLim(1,2)+1)*nsm
      meshl(3) = (meshLim(2,3) - meshLim(1,3)+1)*nsm

C     Calculate starting point for grid

      Xoffset = (meshLim(1,1)-1)*nsm
      Yoffset = (meshLim(1,2)-1)*nsm
      Zoffset = (meshLim(1,3)-1)*nsm

      
C     Add the electric field potential to the input potential

      imesh = 0
      i30 = Zoffset - 1
      do i3 = 1,meshl(3)
         i30 = i30 + 1
         j3 = mod( i30-i0(3), meshG(3) )
         i20 = Yoffset - 1
         do i2 = 1,meshl(2)
            i20 = i20 + 1
            j2 = mod( i20-i0(2), meshG(2) )
            i10 = Xoffset - 1
            do i1 = 1,meshl(1)
               i10 = i10 + 1
               j1 = mod( i10-i0(1), meshG(1) )
               imesh = imesh + 1
               v(imesh) = v(imesh) +
     &              v0 + f(1)*j1 + f(2)*j2 + f(3)*j3
            enddo
         enddo
      enddo

      end subroutine add_potential_from_field

      end module m_efield
