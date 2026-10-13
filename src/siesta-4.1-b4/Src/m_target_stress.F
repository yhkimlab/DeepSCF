! ---
! Copyright (C) 1996-2016	The SIESTA group
!  This file is distributed under the terms of the
!  GNU General Public License: see COPYING in the top directory
!  or http://www.gnu.org/copyleft/gpl.txt .
! See Docs/Contributors.txt for a list of contributors.
! ---
!
      module m_target_stress
!
!     Handles the implementation of a target stress for geometry
!     optimization.
!
      use precision, only : dp

      real(dp), private, save :: target_stress(3,3)
      logical, private, save :: constant_volume

      public :: set_target_stress
      public :: subtract_target_stress

      CONTAINS

      subroutine set_target_stress()

c   It allows an external target stress:
c              %block MD.TargetStress
c                  3.5  0.0  0.0  0.0  0.0  0.0
c              %endblock MD.TargetStress
c   corresponding to xx, yy, zz, xy, xz, yz.
c   In units of (-MD.TargetPressure)
c   Default: hydrostatic pressure: -1, -1, -1, 0, 0, 0
c
      use parallel,    only : Node
      use sys,         only : die
      use fdf
      use units, only: kBar
!
      implicit none

c Internal variables and arrays

      real(dp) :: trace, tp
      integer  :: i, j

      real(dp) ::  sxx, syy, szz, sxy, sxz, syz

      type(block_fdf)            :: bfdf
      type(parsed_line), pointer :: pline=>null()

      logical   :: tarstr = .false.

         ! Check if we want a constant-volume simulation
         ! Set scale for target stress
!
         tp = fdf_get('MD.TargetPressure',0.0_dp,'Ry/Bohr**3')
         constant_volume = fdf_get("MD.ConstantVolume", .false.)

C Look for target stress and read it if found, otherwise generate it --------

          tarstr = fdf_block('MD.TargetStress',bfdf)

          if (tarstr) then
            if (Node.eq.0) then
              write(6,'(/a,a)') 'Reading %block MD.TargetStress',
     .                          ' (units of MD.TargetPressure).'
            endif
            if (.not. fdf_bline(bfdf,pline))
     .        call die('ERROR in MD.TargetStress block')
            sxx = fdf_bvalues(pline,1)
            syy = fdf_bvalues(pline,2)
            szz = fdf_bvalues(pline,3)
            sxy = fdf_bvalues(pline,4)
            sxz = fdf_bvalues(pline,5)
            syz = fdf_bvalues(pline,6)
            call fdf_bclose(bfdf)
            
            target_stress(1,1) = - sxx * tp
            target_stress(2,2) = - syy * tp
            target_stress(3,3) = - szz * tp
            target_stress(1,2) = - sxy * tp
            target_stress(2,1) = - sxy * tp
            target_stress(1,3) = - sxz * tp
            target_stress(3,1) = - sxz * tp
            target_stress(2,3) = - syz * tp
            target_stress(3,2) = - syz * tp
          else
            if (Node.eq.0) then
              write(6,'(/a,a)') 'No target stress found, ',
     .             'assuming hydrostatic MD.TargetPressure.'
            endif
            do i= 1, 3
              do j= 1, 3
                target_stress(i,j) = 0._dp
              enddo
              target_stress(i,i) = - tp
            enddo
          endif

C Write target stress down --------------------------------------------------

          if (constant_volume) then
            target_stress(:,:) = 0.0_dp
            if (Node.eq.0) then
              write(6,"(a)") "***Target stress set to zero " //
     $            "for constant-volume calculation"
            endif
          endif
          if (Node.eq.0) then
            write(6,"(/a)") 'Target stress (kBar)'
            write(6,"(a,2x,3f12.3)") 
     .       ' ', target_stress(1,1)/kBar, target_stress(1,2)/kBar,
     .       target_stress(1,3)/kBar
            write(6,"(a,2x,3f12.3)") 
     .       ' ', target_stress(2,1)/kBar, target_stress(2,2)/kBar,
     .       target_stress(2,3)/kBar
            write(6,"(a,2x,3f12.3)") 
     .       ' ', target_stress(3,1)/kBar, target_stress(3,2)/kBar,
     .       target_stress(3,3)/kBar
          endif


      end subroutine set_target_stress
!
!---------------------------------------------------------------
      subroutine subtract_target_stress(stress_in,stress)
      !
      ! Removes the target_stress from stress_in
      ! and stores the result in stress

      real(dp), intent(in)  :: stress_in(3,3)
      real(dp), intent(out) :: stress(3,3)

      real(dp) :: trace
      integer :: i, j

      stress = stress_in

      ! First, symmetrize
        do i = 1, 3
           do j = i+1, 3
              stress(i,j) = 0.5_dp*( stress(i,j) + stress(j,i) )
              stress(j,i) = stress(i,j)
           enddo
        enddo

!        Subtract target stress

        stress = stress - target_stress
!
!       Take 1/3 of the trace out here if constant-volume needed
!
        if (constant_volume) then
           trace = stress(1,1) + stress(2,2) + stress(3,3)
           do i=1,3
              stress(i,i) = stress(i,i) - trace/3.0_dp
           enddo
        endif

      end subroutine subtract_target_stress

      end module m_target_stress


