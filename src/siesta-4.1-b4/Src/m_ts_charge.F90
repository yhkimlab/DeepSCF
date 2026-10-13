!
! Copyright (C) 1996-2016	The SIESTA group
!  This file is distributed under the terms of the
!  GNU General Public License: see COPYING in the top directory
!  or http://www.gnu.org/copyleft/gpl.txt.
! See Docs/Contributors.txt for a list of contributors.
!
! This code segment has been fully created by:
! Nick Papior Andersen, 2013, nickpapior@gmail.com
! Please conctact the author, prior to re-using this code.
! * It has been heavily inspired by the original authors of the 
!   Transiesta code (hence the references here are still remaining) *

! Module for correcting the density matrix for retaining a constant charge density
! The idea is to introduce several different schemes of charge corrections.

module m_ts_charge

  use precision, only: dp
  
  implicit none

  public 

  ! Info parameters for obtaining charge calculations (mulliken charges in regions)
  integer, parameter :: TS_INFO_FULL = 0
  integer, parameter :: TS_INFO_SCF = 1

  ! Method parameters for the charge-correction
  integer, save :: TS_RHOCORR_METHOD = 0
  integer, parameter :: TS_RHOCORR_BUFFER = 1
  integer, parameter :: TS_RHOCORR_FERMI = 2
  real(dp), save :: TS_RHOCORR_FERMI_TOLERANCE = 0.01_dp
  real(dp), save :: TS_RHOCORR_FERMI_MAX = 0.1102471_dp ! 1.5 eV
  real(dp), save :: TS_RHOCORR_FACTOR = 0.75_dp

  private :: dp

contains

  subroutine read_ts_charge_cor( )
    
    use fdf, only : fdf_get, leqi
    character(len=200) :: chars
    
    chars = fdf_get('TS.ChargeCorrection','none')
    TS_RHOCORR_METHOD = 0
    if ( leqi(chars,'none') ) then
       TS_RHOCORR_METHOD = 0
    else if ( leqi(chars,'b') .or. leqi(chars,'buffer') ) then
       TS_RHOCORR_METHOD = TS_RHOCORR_BUFFER
    else if ( leqi(chars,'fermi') ) then
       TS_RHOCORR_METHOD = TS_RHOCORR_FERMI
    end if
    TS_RHOCORR_FERMI_TOLERANCE = &
         fdf_get('TS.ChargeCorrection.Fermi.Tolerance',0.01_dp)
    ! Factor for charge-correction
    TS_RHOCORR_FACTOR = fdf_get('TS.ChargeCorrection.Factor',0.75_dp)
    if ( 1.0_dp < TS_RHOCORR_FACTOR ) then
       call die("Charge correction factor must be in the range [0;1]")
    endif
    if ( TS_RHOCORR_FACTOR < 0.0_dp ) then
       call die("Charge correction factor must be larger than 0")
    endif
    ! Truncation of fermi-level change (default 1.5 eV)
    TS_RHOCORR_FERMI_MAX = fdf_get('TS.ChargeCorrection.Fermi.Max',0.1102471_dp,'Ry')

  end subroutine read_ts_charge_cor

  ! Retrive the mulliken charges in each region of the transiesta setup
  subroutine ts_get_charges(N_Elec,dit, sp, nspin, n_nzs, DM, S, Q, Qtot)

    use m_ts_method
    use parallel, only : Node
#ifdef MPI
    use mpi_siesta
#endif
    use class_OrbitalDistribution
    use class_Sparsity
    use geom_helper, only : UCORB
    use m_ts_electype

! **********************
! * INPUT variables    *
! **********************
    integer, intent(in) :: N_Elec
    type(OrbitalDistribution), intent(inout) :: dit
    ! SIESTA local sparse pattern (not changed)
    type(Sparsity), intent(inout) :: sp
    ! Number of non-zero elements
    integer, intent(in) :: nspin, n_nzs
    ! The density matrix and overlap
    real(dp), intent(in) :: DM(n_nzs,nspin), S(n_nzs)
    ! The charge in the regions
    real(dp), intent(out), optional :: Q(0:1+1+N_Elec*2, nspin), Qtot
    
! **********************
! * LOCAL variables    *
! **********************
    integer, pointer :: l_ncol(:), l_ptr(:), l_col(:)
    integer :: no_lo, no_u, lio, io, ind, jo, ir, jr, r
    real(dp) :: Qtmp(0:1+1+N_Elec*2, nspin)
#ifdef MPI
    real(dp) :: tmp(0:1+1+N_Elec*2, nspin)
    integer :: MPIerror
#endif

    ! Retrieve information about the sparsity pattern
    call attach(sp, &
         n_col=l_ncol,list_ptr=l_ptr,list_col=l_col, &
         nrows=no_lo,nrows_g=no_u)
    
    ! Initialize charges
    Qtmp(:,:) = 0._dp

!$OMP parallel do default(shared), &
!$OMP&private(lio,io,ir,ind,jo,jr,r), &
!$OMP&reduction(+:Qtmp)
    do lio = 1 , no_lo

       ! obtain the global index of the orbital.
       io = index_local_to_global(dit,lio,Node)
       ir = orb_type(io)

       ! Loop number of entries in the row... (index frame)
       do ind = l_ptr(lio) + 1 , l_ptr(lio) + l_ncol(lio)
          
          ! as the local sparsity pattern is a super-cell pattern,
          ! we need to check the unit-cell orbital
          ! The unit-cell column index
          jo = UCORB(l_col(ind),no_u)
          jr = orb_type(jo)

          if      ( all((/ir,jr/) == TYP_BUFFER) ) then
             r = 1 ! buffer
          else if ( any((/ir,jr/) == TYP_BUFFER) ) then
             r = 0 ! other
          else if ( all((/ir,jr/) == TYP_DEVICE) ) then
             r = 2 ! device
          else if ( any((/ir,jr/) == TYP_DEVICE) ) then
             r = 4+(ir+jr-1)*2 ! device/electrode
          else if ( ir == jr ) then
             r = 3+(ir-1)*2 ! electrode/electrode
          else
             r = 0 ! other
          end if
          Qtmp(r,:) = Qtmp(r,:) + DM(ind,:) * S(ind)
       end do
    end do
!$OMP end parallel do

#ifdef MPI
    call MPI_AllReduce(Qtmp(0,1),tmp(0,1),size(Qtmp), &
         MPI_Double_Precision,MPI_SUM, MPI_Comm_World,MPIerror)
    if ( present(Q)    ) Q    = tmp
    if ( present(Qtot) ) Qtot = sum(tmp)
#else
    if ( present(Q)    ) Q    = Qtmp
    if ( present(Qtot) ) Qtot = sum(Qtmp)
#endif

  end subroutine ts_get_charges

  ! A subroutine for printing out the charge distribution in the cell
  ! it will currently only handle the full charge distribution, and
  ! not per k-point.
  subroutine ts_print_charges(N_Elec,Elecs,Qtot,dit, sp, &
      nspin, n_nzs, DM, S, &
      method)
    use parallel, only : IONode
    use m_ts_electype
#ifdef MPI
    use mpi_siesta
#endif
    use class_OrbitalDistribution
    use class_Sparsity
    use geom_helper, only : UCORB

    ! **********************
    ! * INPUT variables    *
    ! **********************
    integer, intent(in) :: N_Elec
    type(Elec), intent(in) :: Elecs(N_Elec)
    ! The requested number of electrons in the simulation
    real(dp), intent(in) :: Qtot
    type(OrbitalDistribution), intent(inout) :: dit
    ! SIESTA local sparse pattern (not changed)
    type(Sparsity), intent(inout) :: sp
    ! Number of non-zero elements
    integer, intent(in) :: nspin, n_nzs
    ! The density matrix and overlap
    real(dp), intent(in) :: DM(n_nzs,nspin), S(n_nzs)
    ! The method by which it should be printed out...
    integer, intent(in), optional :: method

    ! **********************
    ! * LOCAL variables    *
    ! **********************
    integer :: i
    real(dp), allocatable :: Q(:,:)
    real(dp) :: sQtot
    integer :: ispin, lmethod
    logical :: has_buffer

    ! Requested charge per spin if anti-ferromagnetic
    sQtot = Qtot / real(nspin,dp)

    lmethod = TS_INFO_FULL
    if ( present(method) ) lmethod = method

    allocate(Q(0:2+N_Elec*2,nspin))

    call ts_get_charges(N_Elec,dit, sp, nspin, n_nzs, DM, S, Q = Q)
    has_buffer = sum(Q(1,:)) > 0._dp

    ! it will only be the IONode which will write out...
    if ( .not. IONode ) return

    if ( lmethod == TS_INFO_FULL ) then
      write(*,'(/,a,f12.5)') 'transiesta: Charge distribution, target = ',Qtot
      if ( nspin > 1 ) then
        write(*,'(a,3(f12.5,tr1))') &
            'Total charge                  [Q]  :', &
            sum(Q(:,1)),sum(Q(:,2)),sum(Q)
        write(*,'(a,2(f12.5,tr1))') &
            'Device                        [D]  :',Q(2,1), Q(2,2)
        do i = 1 , N_Elec
          write(*,'(a,t31,a,i0,a,2(f12.5,tr1))') &
              trim(name(Elecs(i))),'[E',i,'] :', &
              Q(3+(i-1)*2,1), Q(3+(i-1)*2,2)
          write(*,'(a,t22,a,i0,a,2(f12.5,tr1))') &
              trim(name(Elecs(i))),'/ device [C',i,'] :', &
              Q(4+(i-1)*2,1), Q(4+(i-1)*2,2)
        end do
        write(*,'(a,2(f12.5,tr1),/)') &
            'Other                         [O]  :',Q(0,1), Q(0,2)
        if ( has_buffer ) then
          write(*,'(a,2(f12.5,tr1))') &
            'Buffer                        [B]  :',Q(1,1), Q(1,2)
        end if
      else
        write(*,'(a,f12.5)') &
            'Total charge                  [Q]  :', sum(Q(:,1))
        write(*,'(a,f12.5)') &
            'Device                        [D]  :',Q(2,1)
        do i = 1 , N_Elec
          write(*,'(a,t31,a,i0,a,f12.5)') &
              trim(name(Elecs(i)))         ,'[E',i,'] :',Q(3+(i-1)*2,1)
          write(*,'(a,t22,a,i0,a,f12.5)') &
              trim(name(Elecs(i))),'/ device [C',i,'] :',Q(4+(i-1)*2,1)
        end do
        if ( has_buffer ) then
          write(*,'(a,f12.5)') &
            'Buffer                        [B]  :',Q(1,1)
        end if
        write(*,'(a,f12.5,/)') &
            'Other                         [O]  :',Q(0,1)
      end if

    else if ( lmethod == TS_INFO_SCF ) then

      ! We write out the information from the SCF cycle...
      write(*,'(a,1x,a9)',advance='no') 'ts-q:','D'
      do i = 1 , N_Elec
        if ( i > 9 ) then
          write(*,'(1x,a7,i2,1x,a7,i2)',advance='no') 'E',i,'C',i
        else
          write(*,'(1x,a8,i1,1x,a8,i1)',advance='no') 'E',i,'C',i
        end if
      end do
      if ( has_buffer ) then
        write(*,'(1x,a9)',advance='no') 'B'
      end if
      if ( nspin > 1 ) then
        write(*,'(2(1x,a9))') 'dQ','dQtot'
      else
        write(*,'(1x,a9)') 'dQ'
      end if
      do ispin = 1 , nspin
        write(*,'(a,1x,f9.3)',advance='no') 'ts-q:', Q(2,ispin)
        do i = 1 , N_Elec
          write(*,'(2(1x,f9.3))',advance='no') Q(3+(i-1)*2,ispin),Q(4+(i-1)*2,ispin)
        end do
        if ( has_buffer ) then
          write(*,'(1x,f9.3)',advance='no') Q(1,ispin)
        end if
        if ( ispin > 1 .and. ispin == nspin ) then
          write(*,'(2(1x,es9.3e1))') sum(Q(:,ispin)) - sQtot,sum(Q) - Qtot
        else
          write(*,'(1x,es9.3e1)') sum(Q(:,ispin)) - sQtot
        end if
      end do

    end if

    deallocate(Q)

  end subroutine ts_print_charges

  subroutine ts_qc(N_Elec,Elecs, &
       dit, sp, nspin, n_nzs, DM, EDM, S, Qtot, &
       method)

    use m_ts_electype
    use class_OrbitalDistribution
    use class_Sparsity

! **********************
! * INPUT variables    *
! **********************
    integer, intent(in) :: N_Elec
    ! The electrodes
    type(Elec), intent(in) :: Elecs(N_Elec)
    type(OrbitalDistribution), intent(inout) :: dit
    ! SIESTA local sparse pattern (not changed)
    type(Sparsity), intent(inout) :: sp
    ! Number of non-zero elements
    integer, intent(in) :: nspin, n_nzs
    ! The density matrices and overlap
    real(dp), intent(inout) :: DM(n_nzs,nspin), EDM(n_nzs,nspin)
    real(dp), intent(in) :: S(n_nzs)
    real(dp), intent(in) :: Qtot
    integer, intent(in) :: method

    if ( method == TS_RHOCORR_BUFFER ) then
       call ts_qc_buffer(N_Elec,Elecs, &
            dit, sp, nspin, n_nzs, DM, EDM, S, Qtot)
    end if

  end subroutine ts_qc

  subroutine ts_qc_Fermi(dit,sp,nspin,n_nzs,DM,S,Qtot, &
       spDM,Efermi,converged)

    use parallel, only : IONode
    use units, only : eV
    use class_OrbitalDistribution
    use class_Sparsity
    use class_dSpData2D
#ifdef MPI
    use mpi_siesta
#endif

    type(OrbitalDistribution), intent(inout) :: dit
    ! SIESTA local sparse pattern (not changed)
    type(Sparsity), intent(inout) :: sp
    ! Number of non-zero elements
    integer, intent(in) :: nspin, n_nzs
    ! The density matrices and overlap
    real(dp), intent(in) :: DM(n_nzs,nspin), S(n_nzs)
    ! The a single energy-point contribution from which
    ! we calculate the differential contribution to the Fermi-level
    type(dSpData2D), intent(inout) :: spDM
    real(dp), intent(in) :: Qtot
    real(dp), intent(inout) :: Efermi
    logical, intent(out) :: converged

! ******************* Local arrays *******************
    type(Sparsity), pointer :: t_sp
    real(dp), pointer :: tDM(:,:)
    real(dp) :: Q(2)
    integer :: lio, lnr, ind, tind, ispin
    integer, pointer :: l_ptr(:), l_ncol(:), l_col(:)
    integer, pointer :: t_ptr(:), t_ncol(:), t_col(:)
#ifdef MPI
    real(dp) :: Qtmp(2)
    integer :: MPIerror
#endif

    ! Populate the arrays
    call attach(sp, &
         n_col=l_ncol,list_ptr=l_ptr,list_col=l_col, &
         nrows=lnr)
    t_sp => spar(spDM)
    call attach(t_sp, &
         n_col=t_ncol,list_ptr=t_ptr,list_col=t_col, &
         nrows=lio)
    if ( lio /= lnr ) call die('Error in sparsity patterns')
    tDM => val(spDM)

    ! Calculate charge @Fermi level
    Q(:) = 0._dp
!$OMP parallel do default(shared), &
!$OMP&private(lio,ind,ispin,tind), &
!$OMP&reduction(+:Q)
    do lio = 1 , lnr
       
       if ( l_ncol(lio) /= 0 ) then
       
       do ind = l_ptr(lio) + 1 , l_ptr(lio) + l_ncol(lio)

          ! The current charge in the Density matrix
          do ispin = 1 , nspin
             Q(1) = Q(1) + DM(ind,ispin) * S(ind)
          end do

          if ( t_ncol(lio) == 0 ) cycle

          do tind = t_ptr(lio) + 1 , t_ptr(lio) + t_ncol(lio)
             
             if ( t_col(tind) /= l_col(ind) ) cycle

             ! the calculated charge @Fermi-level
             Q(2) = Q(2) + tDM(tind,1) * S(ind)

          end do

       end do

       end if
       
    end do
!$OMP end parallel do

#ifdef MPI
    Qtmp(:) = Q(:)
    call MPI_AllReduce(Qtmp, Q, 2, MPI_Double_Precision, &
         MPI_Sum, MPI_Comm_World, MPIerror)
#endif
    
    ! The additional charge
    Q(1) = Q(1) - Qtot
    if ( abs(Q(1)) < TS_RHOCORR_FERMI_TOLERANCE ) then
       converged = .true.
       if ( IONode ) &
            write(*,'(a,e10.3)') 'ts-qc: Fermi-level correction converged dQ: ',Q(1)
       return
    end if

    ! Now we have the difference in Q
    ! Correct the Fermi level so that a change dE DM would 
    ! account for the missing/excess charge.
    ! dE * DM@(Ef) = dQ => dE = dQ / DM@(Ef)
    Q(2) = Q(1) / Q(2) * TS_RHOCORR_FACTOR

    ! If Ef lies in the middle of bands we will have no DOS
    ! right at the Fermi level.
    ! If this is the case we truncate the change in Fermi-level
    ! to the maximum allowed shift...
    converged = ts_qc_Fermi_truncate(0._dp,TS_RHOCORR_FERMI_MAX,Q(2))

    ! We have not converged
    converged = .false.

    ! As the energy filling must increase for positive
    ! shifting of the Fermi-level we subtract as Q(2) is 
    ! a positive number (for additional charge)
    Efermi = Efermi - Q(2)
    if ( IONode ) then
       write(*,'(a,e11.4,a)') 'ts-qc-iscf: first dEf = ',-Q(2)/eV,' eV'
    end if

  end subroutine ts_qc_Fermi

  subroutine ts_qc_Fermi_file(Ef)

    use parallel, only : Node
    use units, only : eV
    use m_interpolate
#ifdef MPI
    use mpi_siesta
#endif
    ! We read in the TS_FERMI file and estimates the
    ! fermi level by interpolating the largest and 
    ! smallest charges for each iterations Fermi level
    real(dp), intent(inout) :: Ef

    integer :: iu, ioerr, i
    real(dp) :: Ef_neg, Ef_pos, cur(2)
    real(dp), allocatable :: Q_Ef(:,:,:), first_Q(:)
    integer :: N, max_itt, tmp
    character(len=2) :: char2
    integer :: all_sign

#ifdef MPI
    integer :: MPIerror
#endif

    ! First we need to read in the entire 
    ! file and figure out the highest number of iterations
    ! performed.
    if ( Node == 0 ) then

       max_itt = 0
       N = 0
       ! open file
       call io_assign(iu)
       open(unit=iu,file='TS_FERMI',form='formatted', &
            status='old',iostat=ioerr)
       if ( ioerr /= 0 ) then
          call die('The file has not been created, this should not happen')
       end if
       rewind(iu)
       
       ! Figure out number of fermi iterations
       do while ( ioerr /= -1 ) 
          N = N + 1
          read(iu,*) ! # TSiscf line
          read(iu,'(a2,i15)')char2,tmp
          max_itt = max(max_itt,tmp)
          do i = 1 , tmp
             read(iu,*) ! data line
          end do
          read(iu,*,iostat=ioerr) ! empty line
       end do

       ! If only one data point is present,
       ! we cannot do anything...
       if ( N > 1 ) then

       ! First calculated charge for every iteration
       allocate(first_Q(N))
       ! Q_Ef(iteration, [Ef, Q], [negative(Q), positive(Q)])
       allocate(Q_Ef(N,2,2))
       Q_Ef(:,1,:) = 0._dp
       Q_Ef(:,2,1) = huge(1._dp)
       Q_Ef(:,2,2) = -huge(1._dp)

       ! Rewind and read data
       rewind(iu)
       ioerr = 0
       ! Read in the data and move it to Q_Ef
       N = 0
       do while ( ioerr /= -1 ) 
          N = N + 1
          read(iu,*) ! # TSiscf line
          read(iu,'(a2,i15)') char2,tmp

          do i = 1 , tmp
             read(iu,'(2(tr1,e20.10))') cur(:)
             
             ! Get first charge (initial charge)
             if ( i == 1 ) first_Q(N) = cur(2)
             ! Convert to Ry
             cur(1) = cur(1) * eV

             ! Assign min Q
             if ( Q_Ef(N,2,1) > cur(2) ) then
                Q_Ef(N,:,1) = cur(:)
             end if
             ! Assign max Q
             if ( Q_Ef(N,2,2) < cur(2) ) then
                Q_Ef(N,:,2) = cur(:)
             end if
             
          end do

          read(iu,*,iostat=ioerr) ! empty line
       end do

       ! We now have all the peaks calculated previously...
       
       ! Interpolate the new fermi level by using the negative charge
       call interp_spline(N,Q_Ef(1:N,2,1),Q_Ef(1:N,1,1),0._dp,Ef_neg)
       ! Interpolate the new fermi level by using the positive charge
       call interp_spline(N,Q_Ef(1:N,2,2),Q_Ef(1:N,1,2),0._dp,Ef_pos)

       all_sign = 0

       ! We first discard the interpolation that is 
       ! clearly wrong. I.e. the one that decides the wrong
       ! direction, this only makes sense
       ! if all previous tries in estimating the fermi-level
       ! has the same tendency. I.e. if we always have excess charge
       ! then we should use this scheme.
       if ( all(first_Q > 0._dp) ) then
          all_sign = 1
          ! We always have too much charge
          ! If their guesses are clearly wrong, we simply
          ! use the interpolated fermi-level, we need
          ! more iterations to clear out this mess
          if ( Ef_pos > Ef ) Ef_pos = Ef
       else if ( all(first_Q < 0._dp) ) then
          all_sign = -1
          if ( Ef_neg < Ef ) Ef_neg = Ef
       end if

       deallocate(Q_Ef,first_Q)

       ! we take the minimum deviating one... :)
       ! We do not tempt our souls to the Fermi-god...
       select case ( all_sign )
       case ( 0 ) 
          if ( abs(Ef_neg - Ef) > abs(Ef_pos - Ef) ) then
             Ef_neg = Ef
             Ef = Ef + TS_RHOCORR_FACTOR * ( Ef_pos - Ef )
          else
             Ef_pos = Ef
             Ef = Ef + TS_RHOCORR_FACTOR * ( Ef_neg - Ef )
             Ef_neg = Ef_pos
          end if
       case ( 1 ) ! all first ones are positive
                  ! i.e. we always underestimate
          Ef_neg = Ef
          Ef = Ef + TS_RHOCORR_FACTOR * ( Ef_pos - Ef )
       case ( -1 ) ! all first ones are negative
                  ! i.e. we always underestimate
          Ef_pos = Ef
          Ef = Ef + TS_RHOCORR_FACTOR * ( Ef_neg - Ef )
          Ef_neg = Ef_pos
       end select
       
       ! Truncate to the maximum allowed difference
       if ( ts_qc_Fermi_truncate(Ef_neg,TS_RHOCORR_FERMI_MAX,Ef) ) then
          ! do nothing
       end if

       ! If we change the fermi-level, just print-out to the user
       if ( abs(Ef_neg - Ef) > 1.e-6_dp*eV ) then
          write(*,'(a,e11.4,a)') 'ts-qc-scf: cubic spline dEf = ', &
               (Ef-Ef_neg)/eV, ' eV'
       end if
       
       end if ! N > 1

       call io_close(iu)

    end if

#ifdef MPI
    call MPI_Bcast(Ef,1,MPI_Double_Precision, &
         0,MPI_Comm_World, MPIerror)
#endif

  end subroutine ts_qc_Fermi_file

  function ts_qc_Fermi_truncate(Ef,max_diff,Ef_new) result(trunc)
    real(dp), intent(in) :: Ef, max_diff
    real(dp), intent(inout) :: Ef_new
    logical :: trunc

    trunc = .false.
    if ( abs(Ef_new - Ef) > max_diff ) then
       trunc = .true.
       if ( Ef_new - Ef > 0._dp ) then
          Ef_new =   max_diff + Ef
       else
          Ef_new = - max_diff + Ef
       end if
    end if

  end function ts_qc_Fermi_truncate

  subroutine ts_qc_buffer(N_Elec,Elecs, &
       dit, sp, nspin, n_nzs, DM, EDM, S, Qtot)

    use m_ts_electype
    use m_ts_method
    use parallel, only : Node
#ifdef MPI
    use mpi_siesta
#endif
    use class_OrbitalDistribution
    use class_Sparsity
    use geom_helper, only : UCORB

! **********************
! * INPUT variables    *
! **********************
    integer, intent(in) :: N_Elec
    ! The electrodes
    type(Elec), intent(in) :: Elecs(N_Elec)
    type(OrbitalDistribution), intent(inout) :: dit
    ! SIESTA local sparse pattern (not changed)
    type(Sparsity), intent(inout) :: sp
    ! Number of non-zero elements
    integer, intent(in) :: nspin, n_nzs
    ! The density matrix and energy density matrix
    real(dp), intent(inout) :: DM(n_nzs,nspin), EDM(n_nzs,nspin)
    ! The overlap
    real(dp), intent(in) :: S(n_nzs)
    ! Total charge of the system
    real(dp), intent(in) :: Qtot

! **********************
! * LOCAL variables    *
! **********************
    ! The charge in the regions
    real(dp), allocatable :: Q(:,:)
    integer, pointer :: l_ncol(:), l_ptr(:), l_col(:)
    integer :: no_lo, no_u, lio, io, ind, ispin
    real(dp) :: reD, addQ(nspin)

    allocate(Q(0:2+N_Elec*2,nspin))
    
    call ts_get_charges(N_Elec, dit, sp, nspin, n_nzs, DM, S, Q = Q)

    ! Retrieve information about the sparsity pattern
    call attach(sp, &
         n_col=l_ncol,list_ptr=l_ptr,list_col=l_col, &
         nrows=no_lo,nrows_g=no_u)
    
    ! Calculate the density factor for obtaining the correct charge
    ! For the left buffer region
    reD = (Qtot-sum(Q(:,:))) / sum(Q(1,:))
    
    ! immediately deallocate charge
    deallocate(Q)

    addQ(:) = 0.0_dp

    ! Reset to zero if not existing
    if ( no_Buf == 0 ) return

    ! Apply charge-correction factor 
    ! This will reduce "heavy" charge fluctuations and
    ! should guard against this.
    reD = reD * TS_RHOCORR_FACTOR
    
    do ispin = 1 , nspin
       do lio = 1 , no_lo
          
          ! obtain the global index of the orbital.
          io = index_local_to_global(dit,lio,Node)

          if ( orb_type(io) /= TYP_BUFFER ) cycle

          ! Loop number of entries in the row... (index frame)
          do ind = l_ptr(lio) + 1 , l_ptr(lio) + l_ncol(lio)

             if ( orb_type(l_col(ind)) /= TYP_BUFFER ) cycle
             
             DM(ind,ispin) = DM(ind,ispin) * reD + &
                  DM(ind,ispin)
             ! We are not sure to do with the energy matrix, hence we don't do anything
             ! As the energy density matrix is an integral over the 
             ! density times energy I expect this is the "correct" way to introduce 
             ! this...
             EDM(ind,ispin) = EDM(ind,ispin) * reD + &
                  EDM(ind,ispin)
             addQ(ispin) = addQ(ispin) + &
                  DM(ind,ispin) * reD

          end do
       end do
       
    end do

  end subroutine ts_qc_buffer

end module m_ts_charge
