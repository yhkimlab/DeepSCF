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

module m_ts_fullg

  use precision, only : dp

  use m_ts_sparse_helper

  use m_ts_dm_update, only : init_DM
  use m_ts_dm_update, only : update_DM
  use m_ts_dm_update, only : add_Gamma_DM
  
  use m_ts_weight, only : weight_DM

  use m_ts_method, only: orb_offset, no_Buf
  
  implicit none
  
  public :: ts_fullg
  
  private
  
contains
  
  subroutine ts_fullg(N_Elec,Elecs, &
       nq, uGF, nspin, na_u, lasto, &
       sp_dist, sparse_pattern, &
       no_u, n_nzs, &
       Hs, Ss, DM, EDM, Ef, DE_NEGF)

    use units, only : eV, Pi
    use parallel, only : Node, Nodes
#ifdef MPI
    use mpi_siesta
#endif

    use alloc, only : re_alloc, de_alloc

    use class_OrbitalDistribution
    use class_Sparsity
    use class_dSpData1D
    use class_dSpData2D

    use m_ts_electype
    ! Self-energy read
    use m_ts_gf
    ! Self-energy expansion
    use m_ts_elec_se

    use m_ts_options, only : Calc_Forces
    use m_ts_options, only : N_mu, mus

    use m_ts_options, only : IsVolt

    use m_ts_sparse, only : ts_sp_uc, tsup_sp_uc
    use m_ts_sparse, only : ltsup_sp_sc, ltsup_sc_pnt
    use m_ts_sparse, only : sc_off

    use m_ts_cctype
    use m_ts_contour_eq,  only : Eq_E, ID2idx, c2weight_eq
    use m_ts_contour_neq, only : nEq_E, has_cE_nEq
    use m_ts_contour_neq, only : N_nEq_ID, c2weight_neq
    
    use m_iterator

    ! Gf calculation
    use m_ts_full_scat

#ifdef TRANSIESTA_DEBUG
    use m_ts_debug
#endif

! ********************
! * INPUT variables  *
! ********************
    integer, intent(in) :: N_Elec
    type(Elec), intent(inout) :: Elecs(N_Elec)
    integer, intent(in) :: nq(N_Elec), uGF(N_Elec)
    integer, intent(in) :: nspin, na_u, lasto(0:na_u)
    type(OrbitalDistribution), intent(inout) :: sp_dist
    type(Sparsity), intent(inout) :: sparse_pattern
    integer, intent(in)  :: no_u
    integer, intent(in)  :: n_nzs
    real(dp), intent(in) :: Hs(n_nzs,nspin), Ss(n_nzs)
    real(dp), intent(inout) :: DM(n_nzs,nspin), EDM(n_nzs,nspin)
    real(dp), intent(in) :: Ef
    real(dp), intent(inout) :: DE_NEGF

! ****************** Electrode variables *********************
    complex(dp), pointer :: GFGGF_work(:) => null()
! ************************************************************

! ******************* Computational arrays *******************
    integer :: ndwork, nzwork, n_s
    real(dp), pointer :: dwork(:,:)
    complex(dp), allocatable, target :: zwork(:), GF(:)

    ! A local orbital distribution class (this is "fake")
    type(OrbitalDistribution) :: fdist
    ! The Hamiltonian and overlap sparse matrices
    type(dSpData1D) :: spH, spS
    ! local sparsity pattern in local SC pattern
    type(dSpData2D) :: spDM, spDMneq
    type(dSpData2D) :: spEDM ! only used if calc_forces
    ! The different sparse matrices that will surmount to the integral
    ! These two lines are in global update sparsity pattern (UC)
    type(dSpData2D) ::  spuDM
    type(dSpData2D) :: spuEDM ! only used if calc_forces
! ************************************************************

! ******************* Computational variables ****************
    type(ts_c_idx) :: cE
    real(dp)    :: kw
    complex(dp) :: W, ZW
    logical :: eq_full_Gf
! ************************************************************

! ******************** Loop variables ************************
    type(itt1) :: Sp
    integer, pointer :: ispin
    integer :: iEl, iID
    integer :: iE, imu, io, idx
! ************************************************************

! ******************* Miscalleneous variables ****************
    integer :: ierr, no_u_TS, off, no
    real(dp), parameter :: bkpt(3) = (/0._dp,0._dp,0._dp/)
! ************************************************************

#ifdef TRANSIESTA_DEBUG
    call write_debug( 'PRE transiesta mem' )
#endif

    ! Number of supercells (even though its gamma we
    ! can have different schemes...)
    n_s = size(sc_off,dim=2)

    ! Number of orbitals in TranSIESTA
    no_u_TS = no_u - no_Buf

    ! We do need the full GF AND a single work array to handle the
    ! left-hand side of the inversion...
    ! We will provide all work arrays as single dimension arrays.
    ! This will make interfaces more stringent and allow for
    ! re-use in several other places.
    ! However, this comes at the cost of programmer book-keeping.
    ! It should be expected that the work arrays return GARBAGE
    ! on ALL routines, i.e. they are not used for anything other
    ! than, yes, work.

    ! The zwork is needed to construct the LHS for solving: G^{-1} G = I
    ! Hence, we will minimum require the full matrix...
    call UC_minimum_worksize(IsVolt, N_Elec, Elecs, no)
    nzwork = max(no_u_TS ** 2, no)
    allocate(zwork(nzwork),stat=ierr)
    if (ierr/=0) call die('Could not allocate space for zwork')
    call memory('A','Z',nzwork,'transiesta')

    ! We only need a partial size of the Green function
    no = no_u_TS
    do iEl = 1 , N_Elec
       if ( Elecs(iEl)%DM_update == 0 ) then ! no elements in electrode are updated
          no = no - TotUsedOrbs(Elecs(iEl))
       end if
    end do
    ! when bias is needed we need the entire GF column
    ! for all the electrodes (at least some of the contour points needs this)
    if ( IsVolt ) then
       no = max(no,sum(TotUsedOrbs(Elecs)))
    end if
    no = no * no_u_TS
    allocate(GF(no),stat=ierr)
    if (ierr/=0) call die('Could not allocate space for GFpart')
    call memory('A','Z',no,'transiesta')

    no = 0
    do iEl = 1 , N_Elec

       ! This seems stupid, however, we never use the Sigma and
       ! GF at the same time. Hence it will be safe
       ! to have them point to the same array.
       ! When the UC_expansion_Sigma_GammaT is called
       ! first the Sigma is assigned and then 
       ! it is required that prepare_GF_inv is called
       ! immediately (which it is)
       ! Hence the GF must NOT be used in between these two calls!
       io = TotUsedOrbs(Elecs(iEl))
       Elecs(iEl)%Sigma => GF(no+1:no+io**2)
       no = no + io ** 2

    end do

    if ( IsVolt ) then
       ! we need only allocate one work-array for
       ! Gf.G.Gf^\dagger
       call re_alloc(GFGGF_work,1,maxval(TotUsedOrbs(Elecs))**2,routine='transiesta')
    end if

    ! Create the Fake distribution
    ! The Block-size is the number of orbitals, i.e. all on the first processor
    ! Notice that we DO need it to be the SIESTA size.
#ifdef MPI
    call newDistribution(no_u,MPI_COMM_WORLD,fdist,name='TS-fake dist')
#else
    call newDistribution(no_u,-1            ,fdist,name='TS-fake dist')
#endif

    ! The Hamiltonian and overlap matrices (in Gamma calculations
    ! we will not have any phases, hence, it makes no sense to
    ! have the arrays in complex)
    call newdSpData1D(ts_sp_uc,fdist,spH,name='TS spH')
    call newdSpData1D(ts_sp_uc,fdist,spS,name='TS spS')

    ! If we have a bias calculation we need additional arrays.
    ! If not bias we don't need the update arrays (we already have
    ! all information in tsup_sp_uc (spDMu))

    ! Allocate space for global sparsity arrays
    no = max(N_mu,N_nEq_id)
    call newdSpData2D(tsup_sp_uc,no,fdist, spuDM, name='TS spuDM')
    ! assign dwork, this will problably come at the expence of
    ! two full reductions, however, we save some memory...
    ndwork = nnzs(tsup_sp_uc)
    dwork => val(spuDM)
    if ( Calc_Forces ) then
       call newdSpData2D(tsup_sp_uc,N_mu,fdist, spuEDM, name='TS spuEDM')
    end if
    
    if ( IsVolt ) then
       ! Allocate space for update arrays, local sparsity arrays
       call newdSpData2D(ltsup_sp_sc,N_mu,    sp_dist,spDM   ,name='TS spDM')
       call newdSpData2D(ltsup_sp_sc,N_nEq_id,sp_dist,spDMneq,name='TS spDM-neq')
       if ( nnzs(ltsup_sp_sc) > ndwork ) then
          ! only update if this array is larger (should only happen in 
          ! few processor setups
          ndwork = nnzs(ltsup_sp_sc)
          dwork => val(spDMneq)
       end if
       if ( Calc_Forces ) then
          call newdSpData2D(ltsup_sp_sc,N_mu, sp_dist,spEDM  ,name='TS spEDM')
       end if
    end if

    ! Whether we should always calculate the full Green function
    eq_full_Gf = all(Elecs(:)%DM_update /= 0)

    ! start the itterators
    call itt_init  (Sp,end=nspin)
    ! point to the index iterators
    call itt_attach(Sp,cur=ispin)

    do while ( .not. itt_step(Sp) )

       call init_DM(sp_dist, sparse_pattern, &
            n_nzs, DM(:,ispin), EDM(:,ispin), &
            tsup_sp_uc, Calc_Forces)

       ! Include spin factor and 1/\pi
       kw = 1._dp / Pi
       if ( nspin == 1 ) kw = kw * 2._dp

#ifdef TRANSIESTA_TIMING
       call timer('TS_HS',1)
#endif

       ! Work-arrays are for MPI distribution...
       call create_HS(sp_dist,sparse_pattern, &
            Ef, &
            N_Elec, Elecs, no_u, & ! electrodes, SIESTA size
            n_nzs, Hs(:,ispin), Ss, &
            spH, spS, &
            ndwork, dwork(:,1)) ! annoyingly we can't pass the full array!!!!!

#ifdef TRANSIESTA_TIMING
       call timer('TS_HS',2)
#endif

#ifdef TRANSIESTA_TIMING
       call timer('TS_EQ',1)
#endif

       ! ***************
       ! * EQUILIBRIUM *
       ! ***************
       call init_val(spuDM)
       if ( Calc_Forces ) call init_val(spuEDM)
       no = no_u_TS
       do iEl = 1 , N_Elec
          if ( Elecs(iEl)%DM_update == 0 ) then
             no = no - TotUsedOrbs(Elecs(iEl))
          end if
       end do
       iE = Nodes - Node
       cE = Eq_E(iE,step=Nodes) ! we read them backwards
       do while ( cE%exist )

          ! *******************
          ! * prep Sigma      *
          ! *******************
          call read_next_GS(ispin, 1, bkpt, &
               cE, N_Elec, uGF, Elecs, &
               nzwork, zwork, .false., forward = .false. )
          do iEl = 1 , N_Elec
             call UC_expansion(cE, Elecs(iEl), nzwork, zwork, &
                  non_Eq = .false. )
          end do

          ! *******************
          ! * prep GF^-1      *
          ! *******************
          call prepare_invGF(cE, no_u_TS, zwork, &
               N_Elec, Elecs, &
               spH=spH , spS=spS)

#ifdef TS_DEV
io=100+iE+node
open(io,form='unformatted')
write(io) cE%e / eV
write(io) no_u_TS,no_u_TS
write(io) zwork(1:no_u_TS**2) / eV
close(io)
#endif

          ! *******************
          ! * calc GF         *
          ! *******************
          if ( eq_full_Gf ) then
             call calc_GF(cE,no_u_TS, zwork, GF)

#ifdef TS_DEV
io=300+iE+node
open(io,form='unformatted')
write(io) cE%e / eV
write(io) no_u_TS, no_u_TS
write(io) gf(1:no_u_TS**2) * eV
close(io)
#endif

          else
             call calc_GF_part(cE, no_u_TS, &
                  N_Elec, Elecs, &
                  zwork, GF)

          end if
          
          ! ** At this point we have calculated the Green function

          ! ****************
          ! * save GF      *
          ! ****************
          do imu = 1 , N_mu
             if ( cE%fake ) cycle ! in case we implement an MPI communication solution...
             call ID2idx(cE,mus(imu)%ID,idx)
             if ( idx < 1 ) cycle
             
             call c2weight_eq(cE,idx, kw, W ,ZW)
             call add_DM( spuDM, W, spuEDM, ZW, &
                  no_u_TS, no, GF, &
                  N_Elec, Elecs, &
                  DMidx=mus(imu)%ID)
          end do

          ! step energy-point
          iE = iE + Nodes
          cE = Eq_E(iE,step=Nodes) ! we read them backwards
       end do

#ifdef TRANSIESTA_TIMING
       call timer('TS_EQ',2)
#endif

#ifdef MPI
       ! We need to reduce all the arrays
       call MPI_Barrier(MPI_Comm_World,io)
       call timer('TS_comm',1)
       call my_full_G_reduce(spuDM,nzwork*2,zwork,N_mu)
       if ( Calc_Forces ) then
          call my_full_G_reduce(spuEDM,nzwork*2,zwork,N_mu)
       end if
       call timer('TS_comm',2)
#endif

       if ( .not. IsVolt ) then
          call update_DM(sp_dist,sparse_pattern, n_nzs, &
               DM(:,ispin), spuDM, Ef=Ef, &
               EDM=EDM(:,ispin), spEDM=spuEDM, &
               UpSpGlobal = .true.)

          ! The remaining code segment only deals with 
          ! bias integration... So we skip instantly

          cycle

       end if

       ! *****************
       ! * only things with non-Equilibrium contour...
       ! *****************

       ! initialize to zero
       ! local sparsity update patterns
       call init_val(spDM)
       call init_val(spDMneq)
       if ( Calc_Forces ) call init_val(spEDM)

       ! transfer data to local sparsity arrays
       call add_Gamma_DM(spDM,   spuDM, D_dim2=N_mu, &
            spEDM=spEDM, spuEDM=spuEDM, E_dim2=N_mu)

#ifdef TRANSIESTA_TIMING
       call timer('TS_NEQ',1)
#endif

       ! *******************
       ! * NON-EQUILIBRIUM *
       ! *******************

       ! We have the definition of: Gamma = i(\Sigma - \Sigma^\dagger)
       ! (not with one half)
       ! Hence we need to half the contribution for the non-equilibrium
       kw = 0.5_dp * kw

       call init_val(spuDM)
       if ( Calc_Forces ) call init_val(spuEDM)
       iE = Nodes - Node
       cE = nEq_E(iE,step=Nodes) ! we read them backwards
       do while ( cE%exist )

          ! *******************
          ! * prep Sigma      *
          ! *******************
          call read_next_GS(ispin, 1, bkpt, &
               cE, N_Elec, uGF, Elecs, &
               nzwork, zwork, .false., forward = .false. )
          do iEl = 1 , N_Elec
             call UC_expansion(cE, Elecs(iEl), nzwork, zwork, &
                  non_Eq = .true. )
          end do

          ! *******************
          ! * prep GF^-1      *
          ! *******************
          call prepare_invGF(cE, no_u_TS, zwork, &
               N_Elec, Elecs, &
               spH =spH , spS =spS)
          
          ! *******************
          ! * calc GF         *
          ! *******************
          call calc_GF_Bias(cE, no_u_TS, &
               N_Elec, Elecs, &
               zwork, GF)

#ifdef TS_DEV
io = 500 + iE + Node
open(io,form='unformatted')
write(io) cE%e / eV
write(io) no_u_TS, TotUsedOrbs(Elecs(1))
write(io) GF(1:no_u_TS * totusedorbs(Elecs(1))) * eV
write(io) no_u_TS, TotUsedOrbs(Elecs(2))
write(io) GF(no_u_TS*totusedorbs(Elecs(1))+1:no_u_TS * sum(totusedorbs(Elecs))) * eV
close(io)
#endif

          ! ** At this point we have calculated the Green function

          ! ****************
          ! * save GF      *
          ! ****************
          off = 0
          do iEl = 1 , N_Elec
             if ( cE%fake ) cycle ! in case we implement an MPI communication solution

             ! offset and number of orbitals
             no = TotUsedOrbs(Elecs(iEl))

             call GF_Gamma_GF(Elecs(iEl), no_u_TS, no, &
                  Gf(no_u_TS*off+1), zwork, size(GFGGF_work), GFGGF_work)

             ! step to the next electrode position
             off = off + no
                
             do iID = 1 , N_nEq_ID
                
                if ( .not. has_cE_nEq(cE,iEl,iID) ) cycle
                
                call c2weight_neq(cE,iID,kw,W,imu,ZW)

                call add_DM( spuDM, W, spuEDM, ZW, &
                     no_u_TS, no_u_TS, zwork, &
                     N_Elec, Elecs, &
                     DMidx=iID, EDMidx=imu, &
                     eq = .false.)
             end do
          end do

          ! step energy-point
          iE = iE + Nodes
          cE = nEq_E(iE,step=Nodes) ! we read them backwards
       end do

#ifdef TRANSIESTA_TIMING
       call timer('TS_NEQ',2)
#endif

#ifdef MPI
       ! We need to reduce all the arrays
       call MPI_Barrier(MPI_Comm_World,io)
       call timer('TS_comm',1)
       call my_full_G_reduce(spuDM, nzwork*2, zwork, N_nEq_id)
       if ( Calc_Forces ) then
          call my_full_G_reduce(spuEDM, nzwork*2, zwork, N_mu)
       end if
       call timer('TS_comm',2)
#endif

#ifdef TRANSIESTA_TIMING
       call timer('TS_weight',1)
#endif

       ! 1. move from global UC to local SC
       ! 2. calculate the correct contribution by applying the weight
       ! 3. add the density to the real arrays
       call add_Gamma_DM(spDMneq, spuDM, D_dim2=N_nEq_id, &
            spEDM=spEDM,  spuEDM=spuEDM, E_dim2=N_mu)
       
       call weight_DM( N_Elec, Elecs, N_mu, mus, na_u, lasto, &
            sp_dist, sparse_pattern, Ss, &
            spDM, spDMneq, spEDM, n_s, sc_off, DE_NEGF)
       
       call update_DM(sp_dist,sparse_pattern, n_nzs, &
            DM(:,ispin), spDM, Ef=Ef, &
            EDM=EDM(:,ispin), spEDM=spEDM, ipnt=ltsup_sc_pnt)

#ifdef TRANSIESTA_TIMING
       call timer('TS_weight',2)
#endif
       
       ! We don't need to do anything here..

    end do ! spin

    call itt_destroy(Sp)

#ifdef TRANSIESTA_DEBUG
    write(*,*) 'Completed TRANSIESTA SCF'
#endif

!***********************
! CLEAN UP
!***********************

    call delete(spH)
    call delete(spS)

    call delete(spuDM)
    call delete(spuEDM)

    call delete(spDM)
    call delete(spDMneq)
    call delete(spEDM)

    ! We can safely delete the orbital distribution, it is local
    call delete(fdist)

    call memory('D','Z',size(zwork)+size(GF),'transiesta')
    deallocate(zwork,GF)

    ! In case of voltage calculations we need a work-array for
    ! handling the GF.Gamma.Gf^\dagger multiplication
    if ( IsVolt ) then
       call de_alloc(GFGGF_work, routine='transiesta')
    end if
   
#ifdef TRANSIESTA_DEBUG
    call write_debug( 'POS transiesta mem' )
#endif

  end subroutine ts_fullg

  ! Update DM
  ! These routines are supplied for easy update of the update region
  ! sparsity patterns
  ! Note that these routines implement the usual rho(Z) \propto - GF
  subroutine add_DM(DM, DMfact,EDM, EDMfact, &
       no1,no2,GF, &
       N_Elec,Elecs, &
       DMidx, EDMidx, &
       eq)

    use class_Sparsity
    use class_dSpData2D
    use m_ts_electype

    ! The DM and EDM equivalent matrices
    type(dSpData2D), intent(inout) :: DM
    complex(dp), intent(in) :: DMfact
    type(dSpData2D), intent(inout) :: EDM
    complex(dp), intent(in) :: EDMfact
    ! The size of GF
    integer, intent(in) :: no1, no2
    ! The Green function
    complex(dp), intent(in) :: GF(no1,no2)
    integer, intent(in) :: N_Elec
    type(Elec), intent(in) :: Elecs(N_Elec)
    ! the index of the partition
    integer, intent(in) :: DMidx
    integer, intent(in), optional :: EDMidx
    logical, intent(in), optional :: eq

    ! Arrays needed for looping the sparsity
    type(Sparsity), pointer :: s
    integer,  pointer :: l_ncol(:), l_ptr(:), l_col(:)
    real(dp), pointer :: D(:,:), E(:,:)
    integer :: io, ind, nr
    integer :: iu, ju, i1, i2
    logical :: leq, hasEDM

    leq = .true.
    if ( present(eq) ) leq = eq

    ! Remember that this sparsity pattern HAS to be in Global UC
    s => spar(DM)
    call attach(s,n_col=l_ncol,list_ptr=l_ptr,list_col=l_col, &
         nrows=nr)
    D => val(DM)
    hasEDM = initialized(EDM)
    if ( hasEDM ) E => val(EDM)

    i1 = DMidx
    i2 = i1
    if ( present(EDMidx) ) i2 = EDMidx


    if ( hasEDM ) then
       if ( leq ) then

!$OMP parallel do default(shared), &
!$OMP&private(io,iu,ind,ju)
          do io = 1 , nr
             ! Quickly go past the buffer atoms...
             if ( l_ncol(io) /= 0 ) then

             ! The update region equivalent GF part
             iu = io - orb_offset(io)
        
             do ind = l_ptr(io) + 1 , l_ptr(io) + l_ncol(io)
                
                ju = l_col(ind) - orb_offset(l_col(ind)) &
                     - offset(N_Elec,Elecs,l_col(ind))
                
                D(ind,i1) = D(ind,i1) - dimag( GF(iu,ju) * DMfact  )
                E(ind,i2) = E(ind,i2) - dimag( GF(iu,ju) * EDMfact )
                
             end do

             end if
          end do
!$OMP end parallel do
     
       else
!$OMP parallel do default(shared), &
!$OMP&private(io,iu,ind,ju)
          do io = 1 , nr
             if ( l_ncol(io) /= 0 ) then
             iu = io - orb_offset(io)
             do ind = l_ptr(io) + 1 , l_ptr(io) + l_ncol(io)
                ju = l_col(ind) - orb_offset(l_col(ind))
                D(ind,i1) = D(ind,i1) + real( GF(iu,ju) * DMfact  ,dp)
                E(ind,i2) = E(ind,i2) + real( GF(iu,ju) * EDMfact ,dp)
             end do
             end if
          end do
!$OMP end parallel do

       end if
    else

       if ( leq ) then
!$OMP parallel do default(shared), &
!$OMP&private(io,iu,ind,ju)
          do io = 1 , nr
             ! Quickly go past the buffer atoms...
             if ( l_ncol(io) /= 0 ) then

             ! The update region equivalent GF part
             iu = io - orb_offset(io)
             
             do ind = l_ptr(io) + 1 , l_ptr(io) + l_ncol(io)
                
                ju = l_col(ind) - orb_offset(l_col(ind)) &
                     - offset(N_Elec,Elecs,l_col(ind))
                
                D(ind,i1) = D(ind,i1) - dimag( GF(iu,ju) * DMfact )
                
             end do
             end if
          end do
!$OMP end parallel do

       else
!$OMP parallel do default(shared), &
!$OMP&private(io,iu,ind,ju)
          do io = 1 , nr
             if ( l_ncol(io) /= 0 ) then
             iu = io - orb_offset(io)
             do ind = l_ptr(io) + 1 , l_ptr(io) + l_ncol(io)
                ju = l_col(ind) - orb_offset(l_col(ind))
                D(ind,i1) = D(ind,i1) + real( GF(iu,ju) * DMfact ,dp)
             end do
             end if
          end do
!$OMP end parallel do

       end if
    end if

  contains
    
    pure function offset(N_Elec,Elecs,io)
      integer, intent(in) :: N_Elec
      type(Elec), intent(in) :: Elecs(N_Elec)
      integer, intent(in) :: io
      integer :: offset
      offset = sum(TotUsedOrbs(Elecs(:)), &
           MASK=(Elecs(:)%DM_update==0) .and. Elecs(:)%idx_o <= io )
    end function offset
    
  end subroutine add_DM


  ! creation of the GF^{-1}.
  ! this routine will insert the zS-H and \Sigma_{LR} terms in the GF 
  subroutine prepare_invGF(cE, no_u,GFinv, &
       N_Elec, Elecs, spH, spS)

    use class_dSpData1D
    use class_Sparsity
    use m_ts_electype
    use m_ts_cctype, only : ts_c_idx
#ifdef TS_DEV
    use parallel,only:ionode
#endif
    use m_ts_full_scat, only : insert_Self_Energies

    ! the current energy point
    type(ts_c_idx), intent(in) :: cE
    ! Remark that we need the left buffer orbitals
    ! to calculate the actual orbital of the sparse matrices...
    integer, intent(in) :: no_u
    complex(dp), intent(out) :: GFinv(no_u**2)
    integer, intent(in) :: N_Elec
    type(Elec), intent(in) :: Elecs(N_Elec)
    ! The Hamiltonian and overlap sparse matrices
    type(dSpData1D), intent(inout) :: spH,  spS

    ! Local variables
    complex(dp) :: Z
    type(Sparsity), pointer :: sp
    integer, pointer :: l_ncol(:), l_ptr(:), l_col(:)
    real(dp), pointer :: H(:), S(:)
    integer :: io, iu, ind, ioff, nr

#ifdef TS_DEV
logical, save :: hasSaved = .false.
integer :: i
#endif

    if ( cE%fake ) return

#ifdef TRANSIESTA_TIMING
    call timer('TS-prep',1)
#endif

    Z = cE%e
    
    sp => spar(spH)
    H  => val (spH)
    S  => val (spS)

    call attach(sp,n_col=l_ncol,list_ptr=l_ptr,list_col=l_col, &
         nrows_g=nr)

#ifdef TS_DEV    
    if (.not. hasSaved )then
       hasSaved = .true.
       GFinv(1:no_u**2) = dcmplx(0._dp,0._dp)
       do io = 1, no_u
          if ( l_ncol(io) == 0 ) cycle
          ioff = orb_offset(io) - 1
          iu = (io - ioff) * no_u
          do ind = l_ptr(io) + 1 , l_ptr(io) + l_ncol(io) 
             ioff = orb_offset(l_col(ind))
             GFinv(iu+l_col(ind)-ioff) = H(ind)
          end do
       end do
       if (ionode) then
          i = 50
          open(i,form='unformatted')
          write(i) dcmplx(100._dp,100._dp)
          write(i) no_u
          write(i) no_u
          write(i) GFinv(1:no_u**2) / eV
          write(i) no_u
          GFinv(1:no_u**2) = dcmplx(0._dp,0._dp)
          do io = 1, no_u
             if ( l_ncol(io) == 0 ) cycle
             ioff = orb_offset(io) - 1
             iu = (io - ioff) * no_u
             do ind = l_ptr(io) + 1 , l_ptr(io) + l_ncol(io) 
                ioff = orb_offset(l_col(ind))
                GFinv(iu+l_col(ind)-ioff) = S(ind)
             end do
          end do
          write(i) GFinv(1:no_u**2)
          close(i)
       end if
    end if
#endif

!$OMP parallel default(shared), private(io,ioff,iu,ind)

    ! Initialize
!$OMP workshare
    GFinv(1:no_u**2) = dcmplx(0._dp,0._dp)
!$OMP end workshare

    ! We will only loop in the central region
    ! We have constructed the sparse array to only contain
    ! values in this part...
!$OMP do
    do io = 1, nr

       if ( l_ncol(io) /= 0 ) then
       
       ioff = orb_offset(io)
       iu = (io - ioff - 1) * no_u
       
       do ind = l_ptr(io) + 1 , l_ptr(io) + l_ncol(io) 
             
          ioff = orb_offset(l_col(ind))
          
          ! Notice that we transpose S and H back here
          ! See symmetrize_HS_Gamma (H is hermitian)
          GFinv(iu+l_col(ind)-ioff) = Z * S(ind) - H(ind)

       end do

       end if
    end do
!$OMP end do

    do io = 1 , N_Elec
       call insert_Self_Energies(no_u, Gfinv, Elecs(io))
    end do

!$OMP end parallel

#ifdef TRANSIESTA_TIMING
    call timer('TS-prep',2)
#endif

  end subroutine prepare_invGF
   
end module m_ts_fullg
