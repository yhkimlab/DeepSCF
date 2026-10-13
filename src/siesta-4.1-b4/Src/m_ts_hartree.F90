!
! Copyright (C) 1996-2016	The SIESTA group
!  This file is distributed under the terms of the
!  GNU General Public License: see COPYING in the top directory
!  or http://www.gnu.org/copyleft/gpl.txt.
! See Docs/Contributors.txt for a list of contributors.
!
! This code segment has been improved or fully created by:
! Nick Papior Andersen, 2014, nickpapior@gmail.com
!
module m_ts_hartree

! Module for fixing the Hartree potential so that the potential fluctuations
! does not go wild.
! This is necessary to get a stable SCF solution
!
! Created and copyrighted by: Nick Papior Andersen, 2014
! The use of this program is allowed for not-for-profit research only.
! Copy or disemination of all or part of this package is not
! permitted without prior and explicit authorization by the author.
  
  use precision, only : grid_p, dp
  use m_ts_electype
  use m_ts_tdir, only : ts_tidx

  implicit none
  
  private
  save

  ! The idea is to have sub routines in this module to do
  ! various Hartree potential fixes
  public :: read_ts_hartree_options
  public :: ts_init_hartree_fix
  public :: ts_hartree_fix
  public :: ts_hartree_elec

  ! The electrode that provides the basin of the constant potential
  type(Elec), pointer, public :: El => null()

  ! Method employed for fixing the Hartree potential

  ! No fixing
  integer, parameter, public :: TS_HA_NONE = 0
  ! The boundary plane in the lower part of the transport direction
  integer, parameter, public :: TS_HA_PLANE = 1
  ! The boundary plane in the lower part of the first electrode
  integer, parameter, public :: TS_HA_ELEC = 2
  ! The entire box of the first electrode
  integer, parameter, public :: TS_HA_ELEC_BOX = 3

  ! The used method
  integer, public :: TS_HA = TS_HA_NONE

  ! The fraction of the actual fix
  real(dp), public :: Vha_frac = 1._dp

  ! The Hartree offset potential as determined from the electrode calculation
  real(dp), public :: Vha_offset = 0._dp

  ! The grid-index in the transport direction
  ! where we fix the Hartree potential.
  integer :: ha_idx = 1

contains

  subroutine read_ts_hartree_options(N_Elec, Elecs, cell, na_u, xa)

    use fdf, only: fdf_get, leqi

    integer, intent(in) :: N_Elec
    type(Elec), intent(inout), target :: Elecs(N_Elec)
    real(dp), intent(in) :: cell(3,3)
    integer, intent(in) :: na_u
    real(dp), intent(in) :: xa(3,na_u)

    character(len=64) :: c
    integer :: iEl

    if ( ts_tidx > 0 ) then
       
       c = fdf_get('TS.Hartree.Fix','plane')

    else
       
       c = fdf_get('TS.Hartree.Fix','elec-plane')

    end if

    if ( leqi(c,'plane') .and. ts_tidx > 0 ) then
       TS_HA = TS_HA_PLANE
    else if ( leqi(c,'elec-plane') .or. leqi(c,'elec') ) then
       TS_HA = TS_HA_ELEC
    else if ( leqi(c,'elec-box') ) then
       TS_HA = TS_HA_ELEC_BOX
    else
       call die('TS.Hartree.Fix: invalid option with respect &
            &to simulation cell. "plane" not allowed for N/=2 or&
            & non-aligned electrodes.')
    end if

    ! Determine the fractional correction of the Hartree fix
    Vha_frac = fdf_get('TS.Hartree.Fix.Frac',1._dp)
    Vha_offset = fdf_get('TS.Hartree.Offset',0._dp, 'Ry')

    ! In case the user has not requested a specific electrode plane
    ! we will find the electrode with the largest plane
    nullify(El)
    c = fdf_get('TS.Hartree.Fix.Elec','largest volume/area electrode')
    do iEl = 1, N_Elec
      if ( leqi(Elecs(iEl)%name, c) ) then
        El => Elecs(iEl)
        exit
      end if
    end do

    if ( .not. associated(El) ) then
      ! Find the "biggest" electrode
      call ts_hartree_elec( N_Elec, Elecs , cell, na_u, xa )
    end if

  end subroutine read_ts_hartree_options

  ! Find the biggest electrode by comparing
  ! either the plane or the volume of the electrode cells
  subroutine ts_hartree_elec(N_Elec, Elecs, cell, na_u, xa)
    
    use intrinsic_missing, only: VNORM
    use m_ts_electype, only: Elec_frac
    
    integer, intent(in) :: N_Elec
    type(Elec), intent(inout), target :: Elecs(N_Elec)
    real(dp), intent(in) :: cell(3,3)
    integer, intent(in) :: na_u
    real(dp), intent(in) :: xa(3,na_u)

    integer :: iE
    real(dp) :: area, tmp, f1, f2, r3(3)
    real(dp), external :: volcel

    ! if the electrode has already been
    ! associated, we will not update it
    if ( associated(El) ) return

    if ( TS_HA == TS_HA_PLANE ) then

       ! The hartree plane can only be used for 1 and 2
       ! electrodes
       if ( N_Elec == 1 ) then
          
          El => Elecs(1)
          return
          
       end if

       ! The plane can only be chosen with
       ! 2 electrodes
       if ( N_Elec /= 2 ) then
          call die('ts_hartree: Error in fixation of plane.')
       end if
       
       ! Select the electrode which is more than a bond length
       ! from its boundary
       call Elec_frac(Elecs(1),cell,na_u,xa,ts_tidx,fmin=f1,fmax=f2)
       call Elec_frac(Elecs(2),cell,na_u,xa,ts_tidx,fmin=tmp,fmax=area)

       if ( f1 < tmp ) then
          ! 1 lie closests to the lower cell-boundary
          ! check which of f1 or 1-area is bigger
          if ( f1 < 1._dp - area ) then
             El => Elecs(2)
          else
             El => Elecs(1)
          end if
       else
          ! 2 lie closests to the lower cell-boundary
          ! check which of tmp or 1-f2 is bigger
          if ( tmp < 1._dp - f2 ) then
             El => Elecs(1)
          else
             El => Elecs(2)
          end if
       end if
       
       return
       
    end if

    ! Easy determination of largest basal plane of electrodes
    ! Note we scale according to the Bloch expansion
    ! as this is necessary!
    area = -1._dp
    do iE = 1 , N_Elec

      if ( TS_HA == TS_HA_ELEC ) then
        
        ! calculate area of plane by non-semi-inf vectors
        select case ( Elecs(iE)%t_dir )
        case ( 1 )
          call cross(Elecs(iE)%cell(:,2), Elecs(iE)%cell(:,3), r3)
          tmp = VNORM(r3) * product(Elecs(iE)%Bloch)
        case ( 2 )
          call cross(Elecs(iE)%cell(:,1), Elecs(iE)%cell(:,3), r3)
          tmp = VNORM(r3) * product(Elecs(iE)%Bloch)
        case ( 3 )
          call cross(Elecs(iE)%cell(:,1), Elecs(iE)%cell(:,2), r3)
          tmp = VNORM(r3) * product(Elecs(iE)%Bloch)
        case ( 4 ) ! B-C
          ! Here there are two planes possible, the plane for the non-semi-infinite
          ! direction and B, C, respectively.
          call cross(Elecs(iE)%cell(:,1), Elecs(iE)%cell(:,2), r3)
          tmp = VNORM(r3)
          call cross(Elecs(iE)%cell(:,1), Elecs(iE)%cell(:,3), r3)
          tmp = max(tmp, VNORM(r3))
        case ( 5 ) ! A-C
          call cross(Elecs(iE)%cell(:,2), Elecs(iE)%cell(:,1), r3)
          tmp = VNORM(r3)
          call cross(Elecs(iE)%cell(:,2), Elecs(iE)%cell(:,3), r3)
          tmp = max(tmp, VNORM(r3))
        case ( 6 ) ! A-B
          call cross(Elecs(iE)%cell(:,3), Elecs(iE)%cell(:,1), r3)
          tmp = VNORM(r3)
          call cross(Elecs(iE)%cell(:,3), Elecs(iE)%cell(:,2), r3)
          tmp = max(tmp, VNORM(r3))
        end select
        
      else
        
        ! We check with volume of electrode
        tmp = Elecs(iE)%na_used / real(Elecs(iE)%na_u, dp) * product(Elecs(iE)%Bloch)
        tmp = VOLCEL(Elecs(iE)%cell) * tmp
        
      end if

      ! Select based on area
      if ( tmp > area ) then
        area = tmp
        El => Elecs(iE)
      end if
    end do

  end subroutine ts_hartree_elec
  
  subroutine ts_init_hartree_fix(cell, na_u, xa, nmesh, nmeshl)

    use intrinsic_missing, only: VNORM, VEC_PROJ, VEC_PROJ_SCA
    use units, only : Ang
    use m_mesh_node, only : offset_r, dMesh, dL
    use parallel, only : IONode
#ifdef MPI
    use mpi_siesta, only : MPI_AllReduce, MPI_Sum, MPI_Barrier
    use mpi_siesta, only : MPI_Comm_World, MPI_integer
#endif

    ! ***********************
    ! * INPUT variables     *
    ! ***********************
    real(dp),   intent(in) :: cell(3,3)
    integer,    intent(in) :: na_u
    real(dp),   intent(in) :: xa(3,na_u)
    integer,    intent(in) :: nmesh(3), nmeshl(3)

    integer :: i1, i2, i3, nlp
    real(dp) :: ll(3), llZ(3), llYZ(3), rcell(3,3), rtmp
#ifdef MPI
    integer :: MPIerror
#endif

    ! Quick skip if not fixing
    if ( TS_HA == TS_HA_NONE ) return

    if ( TS_HA == TS_HA_PLANE ) then

       call reclat(cell, rcell, 0) ! without 2pi
       
       ! Calculate the index where we will fix
       ! the Hartree potential.

       ! 'El' points to the electrode which has the
       ! cell boundary farthest from the atoms

       ! Calculate the fraction point where we will cut
       call Elec_frac(El,cell,na_u,xa,ts_tidx, &
            fmin = llYZ(1), fmax = llYZ(2) )

       ! Unit-cell vector
       ll = cell(:,ts_tidx)
       ll = ll / VNORM(ll)

       if ( llYZ(1) < 1._dp - llYZ(2) ) then
          
          ! We have a lower index point
          ! Correct fraction by the interlayer distance
          llYZ(1) = llYZ(1) - &
               sum(ll*El%dINF_layer*0.5_dp*rcell(:,ts_tidx))
          
       else

          llYZ(1) = llYZ(2) + &
               sum(ll*El%dINF_layer*0.5_dp*rcell(:,ts_tidx))

       end if

       ! Figure out the index
       ha_idx = nint(llYZ(1) * nmesh(ts_tidx))
       ha_idx = max(1,ha_idx)
       ha_idx = min(nmesh(ts_tidx),ha_idx)

       if ( IONode ) then
         write(*,*)
         write(*,'(3a)')'ts: Using electrode: ',trim(El%Name), &
              ' for Hartree correction'
         write(*,'(a,f8.5)')'ts: Grid fraction plane ',llYZ(1)
         write(*,'(a,3(tr1,f13.5))') 'ts: Grid point plane (Ang):',&
              llYZ(1)*cell(:,ts_tidx)/Ang
       end if
    
       return
       
    end if

    ! We now were to put the Hartree correction
    if ( TS_HA /= TS_HA_ELEC .and. &
         TS_HA /= TS_HA_ELEC_BOX ) return

    ! We check that we actually process something...
    nlp = 0
    select case ( TS_HA )
    case ( TS_HA_ELEC )
!$OMP parallel do default(shared), &
!$OMP&private(i3,i2,i1,llZ,llYZ,ll), &
!$OMP&reduction(+:nlp)
       do i3 = 0 , nmeshl(3) - 1
          llZ(:) = offset_r(:) + i3*dL(:,3)
          do i2 = 0 , nmeshl(2) - 1
             llYZ(:) = i2*dL(:,2) + llZ(:)
             do i1 = 0 , nmeshl(1) - 1
                ll(:) = i1*dL(:,1) + llYZ(:)
                if ( in_basal_Elec(El%p,ll,dMesh) ) then
                   nlp = nlp + 1
                end if
             end do
          end do
       end do
!$OMP end parallel do
    case ( TS_HA_ELEC_BOX )
!$OMP parallel do default(shared), &
!$OMP&private(i3,i2,i1,llZ,llYZ,ll), &
!$OMP&reduction(+:nlp)
       do i3 = 0 , nmeshl(3) - 1
          llZ(:) = offset_r(:) + i3*dL(:,3)
          do i2 = 0 , nmeshl(2) - 1
             llYZ(:) = i2*dL(:,2) + llZ(:)
             do i1 = 0 , nmeshl(1) - 1
                ll(:) = i1*dL(:,1) + llYZ(:)
                if ( in_Elec(El%box,ll,dMesh) ) then
                   nlp = nlp + 1
                end if
             end do
          end do
       end do
!$OMP end parallel do
    end select

#ifdef MPI
    call MPI_AllReduce(nlp,i1,1,MPI_integer,MPI_Sum, &
         MPI_Comm_World,MPIerror)
    nlp = i1
#endif

    if ( IONode ) then
       write(*,*)
       write(*,'(3a)')   'ts: Using electrode: ',trim(El%Name),' for Hartree correction'
       write(*,'(a,i0)') 'ts: Number of points used: ',nlp
       if ( nlp == 0 ) then
          write(*,'(a)') 'ts: Basal plane of electrode '// &
               trim(El%name)//' might be outside of &
               &unit cell.'
          write(*,'(a)') 'ts: Please move structure so this point is &
               &inside unit cell (Ang):'
          write(*,'(a,3(tr1,f13.5))') 'ts: Plane, point in plane (Ang):', El%p%c / Ang
          write(*,'(a,3(tr1,f13.5))') 'ts: Plane, normal vector  (Ang):', El%p%n
          if ( El%t_dir <= 3 ) then
            write(*,'(a,3(tr1,f13.5))') 'ts: Projected point onto semi-infinite direction (Ang):', &
                VEC_PROJ(cell(:,El%pvt(El%t_dir)), El%p%c) / Ang
            write(*,'(a)') 'ts: The following block will most likely be usable (otherwise try different displacements)'
            
            write(*,'(/,a)') '%block AtomicCoordinatesOrigin'
            rtmp = VEC_PROJ_SCA(cell(:,El%pvt(El%t_dir)), El%p%c) / VNORM(cell(:,El%pvt(El%t_dir)))
            if ( rtmp > 1._dp ) then
              rtmp = rtmp - 1._dp
              write(*,'(tr2,3(tr1,f13.5))') - cell(:,El%pvt(El%t_dir)) * rtmp / Ang
            else if ( rtmp < 0._dp ) then
              write(*,'(tr2,3(tr1,f13.5))') cell(:,El%pvt(El%t_dir)) * rtmp / Ang
            else
              ! The point is most probably very close to the boundary
              ! So shift it
              rtmp = El%dINF_layer * 0.5_dp / VNORM(cell(:,El%pvt(El%t_dir)))
              write(*,'(tr2,3(tr1,f13.5))') - cell(:,El%pvt(El%t_dir)) * rtmp / Ang
            end if
            write(*,'(a,/)') '%endblock'
          end if
       end if
    end if

    if ( nlp == 0 ) then
#ifdef MPI
       call MPI_Barrier(MPI_Comm_World, MPIerror)
#endif
       call die('The partitioning of the basal plane went wrong. &
            &No points are encapsulated.')
    end if

  end subroutine ts_init_hartree_fix

#ifdef INTEL_COMPILER_ERROR 
! This code segment will create an error with the
! intel compiler compiled at a high setting: TODO (not with Gfortran)
!  -m64 -O3 -xHost -fp-model source -fp-model except -ip -prec-div -prec-sqrt
! If I remove the pointer/target feature below it works beautifully!
! *** NOT GOOD ***
  integer, target :: i10, i20, i30
  integer, pointer :: iT
  if ( ts_tidx == 1 ) then
     i10 = 0
     iT => i10
  else if ( ts_tidx == 2 ) then
     i20 = 0
     iT => i20
  else if ( ts_tidx == 3 ) then
     i30 = 0
     iT => i30
  else
     call die('Hartree fix, not implemented')
  end if

  i10 = 0
  i20 = offset_i(2) - 1
  i30 = offset_i(3) - 1
  if ( iT <= 0 ) then
     imesh = 0
     i30 = offset_i(3) - 1
     do i3 = 0,nmeshl(3)-1
        i30 = i30 + 1
        i20 = offset_i(2) - 1
        do i2 = 0,nmeshl(2)-1
           i20 = i20 + 1
           do i10 = 0,nmeshl(1)-1
              imesh = imesh + 1
              if (iT.eq.0) then
                 nlp = nlp + 1
                 Vtot = Vtot + Vscf(imesh)
              end if
           end do
        end do
     end do
  end if
#endif

  ! Fix the potential
  subroutine ts_hartree_fix(nmesh, nmeshl, Vscf)

    use parallel, only: IONode
    use units, only: eV
#ifdef MPI
    use mpi_siesta, only : MPI_AllReduce, MPI_Sum
    use mpi_siesta, only : MPI_Comm_World, MPI_integer
    use mpi_siesta, only : MPI_double_precision
#endif
    use m_mesh_node, only: mesh_correct_idx
    use m_mesh_node, only : offset_i, offset_r, dMesh, dL

    integer, intent(in) :: nmesh(3), nmeshl(3)
    real(grid_p), intent(inout) :: Vscf(:)

    ! Internal variables
    integer :: i1 , i2 , i3, i
    integer :: imesh, nlp
    integer :: i10, i20, i30
#ifdef MPI
    integer :: MPIerror
#endif
    real(dp) :: Vav, Vtot
    real(dp) :: ll(3), llZ(3), llYZ(3)
    integer :: imin(3), imax(3), idx(3)

    ! Quick skip if not fixing
    if ( TS_HA == TS_HA_NONE ) return

#ifdef TRANSIESTA_DEBUG
    call write_debug( 'PRE TS_VH_fix' )
#endif

    ! Initialize summation
    Vtot = 0._dp

    ! Initialize counters
    nlp   = 0
    imesh = 0

    select case ( TS_HA )
    case ( TS_HA_PLANE )
       
       ! Calculate index
       i = ha_idx - offset_i(ts_tidx)
       if ( ts_tidx == 1 .and. &
            0 < i .and. i <= nmeshl(1) ) then
          ! jump to correct X
          imesh = i
          do i3 = 1 , nmeshl(3)
             do i2 = 1 , nmeshl(2)
                nlp  = nlp + 1
                Vtot = Vtot + Vscf(imesh)
                ! skip X box
                imesh = imesh + nmeshl(1)
             end do
          end do
       else if ( ts_tidx == 2 .and. &
            0 < i .and. i <= nmeshl(2) ) then
          ! offset to first Y index
          imesh = (i-1)*nmeshl(1)
          do i3 = 1 , nmeshl(3)
             do i1 = 1 , nmeshl(1)
                nlp  = nlp + 1
                imesh = imesh + 1
                Vtot = Vtot + Vscf(imesh)
             end do
             ! jump entire X and Y box to get the next Z
             ! position with the correct Y offset
             ! Note -1 because we already have passed one Y
             ! block
             imesh = imesh + (nmeshl(2)-1)*nmeshl(1)
          end do
       else if ( ts_tidx == 3 .and. &
            0 < i .and. i <= nmeshl(3) ) then
          imesh = (i-1)*nmeshl(1)*nmeshl(2)
          do i2 = 1 , nmeshl(2)
             do i1 = 1 , nmeshl(1)
                nlp  = nlp + 1
                imesh = imesh + 1
                Vtot = Vtot + Vscf(imesh)
             end do
          end do
       else if ( ts_tidx < 1 .or. 3 < ts_tidx ) then
          call die('Unknown ts_idx direction, option erronous')
       end if

    case ( TS_HA_ELEC )
       
       ! This is an electrode averaging...
       do i3 = 0 , nmeshl(3) - 1
          llZ(:) = offset_r(:) + i3*dL(:,3)
          do i2 = 0 , nmeshl(2) - 1
             llYZ(:) = i2*dL(:,2) + llZ(:)
             do i1 = 0 , nmeshl(1) - 1
                ll(:) = i1*dL(:,1) + llYZ(:)
                imesh = imesh + 1
                if ( in_basal_Elec(El%p,ll,dMesh) ) then
                   nlp  = nlp + 1
                   Vtot = Vtot + Vscf(imesh)
                end if
             end do
          end do
       end do
       
    case ( TS_HA_ELEC_BOX )
       
       ! This is an electrode averaging...
       call Elec_box2grididx(El,nmesh,dL,imin,imax)
       
       ! Now we have the minimum index for the box encompassing
       ! the electrode
       
       ! Loop the indices, and figure out whether
       ! each of them lies in the local grid
!$OMP parallel do default(shared), private(i1,i2,i3,idx,imesh), &
!$OMP&collapse(3), reduction(+:nlp)
       do i3 = imin(3) , imax(3)
        do i2 = imin(2) , imax(2)
         do i1 = imin(1) , imax(1)

            ! Transform to local grid-points
            idx = (/i1,i2,i3/)
            
            ! Transform the index to the unit-cell index
            call mesh_correct_idx(nmesh,idx)

            ! Figure out if this index lies
            ! in the current node
            idx(1) = idx(1) - offset_i(1)
            if ( 0 < idx(1) .and. idx(1) <= nmeshl(1) ) then
             idx(2) = idx(2) - offset_i(2) - 1 ! one more for factor
             if ( 0 <= idx(2) .and. idx(2) < nmeshl(2) ) then
              idx(3) = idx(3) - offset_i(3) - 1 ! one more for factor
              if ( 0 <= idx(3) .and. idx(3) < nmeshl(3) ) then
                 ! Calculate position in local mesh
                 imesh = idx(1) + idx(2)*nmeshl(1)
                 imesh = imesh + idx(3)*nmeshl(1)*nmeshl(2)
                 Vtot = Vtot + Vscf(imesh)
                 nlp = nlp + 1
              end if
             end if
            end if
         end do
        end do
       end do
!$OMP end parallel do
       
    case default
       call die('Something went extremely wrong...Hartree Fix')
    end select
    
    ! Scale the contribution
    Vtot = Vtot * Vha_frac

#ifdef MPI
    call MPI_AllReduce(Vtot,Vav,1,MPI_double_precision,MPI_Sum, &
         MPI_Comm_World,MPIerror)
    Vtot = Vav
    call MPI_AllReduce(nlp,i1,1,MPI_integer,MPI_Sum, &
         MPI_Comm_World,MPIerror)
    nlp = i1
#endif

    if ( nlp == 0 ) then
       call die('The partitioning of the basal plane went wrong. &
            &No points are encapsulated.')
    end if

    Vav = Vtot / nlp - Vha_offset

    if ( IONode ) then
       write(*,'(a,e12.5,a)')'ts-Vha: ',Vav / eV,' eV'
    end if
    
    ! Align potential
!$OMP parallel workshare default(shared)
    Vscf(:) = Vscf(:) - Vav
!$OMP end parallel workshare

#ifdef TRANSIESTA_DEBUG
    call write_debug( 'POS TS_VH_fix' )
#endif

  end subroutine ts_hartree_fix

end module m_ts_hartree
