! This module will control what is used of the electrodes in the transiesta SCF
!
! Hence we here collect the routines for reading and expanding the self-energies
! in the GF-files.

module m_ts_elec_se

  use precision, only : dp

  use m_ts_electype
  use m_ts_cctype

  implicit none

  private

  public :: UC_minimum_worksize
  public :: UC_expansion
  !public :: UC_expansion_Sigma_Bulk
  !public :: UC_expansion_Sigma
  !public :: UC_expansion_Sigma_GammaT
  public :: update_UC_expansion_A

contains

  !> Determine the minimum worksize required for expanding the SE
  subroutine UC_minimum_worksize(IsVolt, NElec, Elecs, nwork)
    !> Whether this is a bias calculation
    logical, intent(in) :: IsVolt
    !> Number of electrodes
    integer, intent(in) :: NElec
    !> Electrodes
    type(Elec), intent(in) :: Elecs(NElec)
    !> Minimum worksize required by UC_expansion
    integer, intent(out) :: nwork

    ! Local variables
    integer :: iE

    ! Initialize
    nwork = 0
    if ( IsVolt ) then
      do iE = 1, NElec
        nwork = max(nwork, TotUsedOrbs(Elecs(iE)) ** 2 * 2)
      end do
    else
      do iE = 1, NElec
        if ( Elecs(iE)%Bulk ) then
          nwork = max(nwork, TotUsedOrbs(Elecs(iE)) ** 2)
        else
          nwork = max(nwork, TotUsedOrbs(Elecs(iE)) ** 2 * 2)
        end if
      end do
    end if

  end subroutine UC_minimum_worksize
  
  subroutine UC_expansion(cE, El, nwork, work, non_Eq)
! ********************
! * INPUT variables  *
! ********************
    type(ts_c_idx), intent(in) :: cE
    type(Elec), intent(in out) :: El

! ********************
! * WORK variables   *
! ********************
    integer,  intent(in) :: nwork
    complex(dp), intent(inout) :: work(nwork)
    
    logical,  intent(in), optional :: non_Eq
    
    complex(dp) :: E
    integer :: nou, no, nq
    logical :: lnon_Eq
    
    if ( cE%fake ) return
    
    call timer('ts_expand',1)
    
    nou = El%no_used
    no  = TotUsedOrbs(El)
    nq  = product(El%Bloch)
    if ( El%pre_expand > 0 .and. nq > 1 ) then
      nou = no
      nq = 1
    end if
    
    ! Save energy
    E = cE%e
    
    lnon_Eq = .false.
    if ( present(non_Eq) ) lnon_Eq = non_Eq
    
    if ( lnon_Eq ) then
#ifdef TBT_PHONON
      E = dcmplx(real(cE%e,dp)**2,El%Eta)
#else
      E = dcmplx(real(cE%e,dp),El%Eta)
#endif
      call UC_expansion_Sigma_GammaT(E, &
          nou,no,El,nq, &
          El%GA,El%Sigma,El%Gamma,nwork,work)
    else
      if ( El%Bulk ) then
        call UC_expansion_Sigma_Bulk(nou,no,El,nq, &
            El%GA,El%Sigma,nwork,work)
      else
        if ( cE%idx(1) /= 1 ) then ! .not. CONTOUR_EQ
#ifdef TBT_PHONON
          E = dcmplx(real(cE%e,dp)**2,El%Eta)
#else
          E = dcmplx(real(cE%e,dp),El%Eta)
#endif
        end if
        call UC_expansion_Sigma(E,nou,no,El,nq, &
            El%GA,El%Sigma,nwork,work)
      end if
    end if

    call timer('ts_expand',2)

  end subroutine UC_expansion

  subroutine UC_expansion_Sigma_Bulk(no_u,no_s,El,nq,&
      GS,Sigma,nwork,work)
    use intrinsic_missing, only : EYE
    use units, only : Pi
! ********************
! * INPUT variables  *
! ********************
    integer,  intent(in) :: no_u, no_s
    type(Elec), intent(in) :: El
    integer,  intent(in) :: nq
    complex(dp), dimension(no_u,no_u,nq), intent(inout) :: GS
! ********************
! * OUTPUT variables *
! ********************
    complex(dp), intent(inout) :: Sigma(no_s,no_s)

    integer,  intent(in) :: nwork
    complex(dp), intent(inout) :: work(no_s,no_s)
! ********************
! * LOCAL variables  *
! ********************
    integer :: ierr
    integer :: ipvt(no_s)

    if ( nq == 1 ) then
#ifndef TS_NOCHECKS
      if ( no_u /= no_s ) call die('no_E/=no_s')
#endif

      ! When no repetition we save it "as is"
      call zcopy(no_s*no_s,GS(1,1,1),1,Sigma(1,1),1)

    else
      
#ifndef TS_NOCHECKS
      if ( nwork < no_s ** 2 ) &
          call die('elec_se-Sigma-Bulk: worksize too small!')
#endif

      if ( El%no_u /= El%no_used ) then

        call update_UC_expansion_A(no_u,no_s,El,nq, &
            El%na_used,El%lasto_used,GS,work(1,1))

        call EYE(no_s,Sigma)

        ! We have the matrix to invert in the first no_s**2 values.
        call zgesv(no_s,no_s,work(1,1),no_s,ipvt,Sigma,no_s,ierr)
        if ( ierr /= 0 ) &
            write(*,'(a,i0)') &
            'Inversion of surface Green function failed: ',ierr

      else

        call update_UC_expansion_A(no_u,no_s,El,nq, &
            El%na_used,El%lasto_used,GS,Sigma)


      end if

    end if

  end subroutine UC_expansion_Sigma_Bulk


  subroutine UC_expansion_Sigma(ZEnergy,no_u,no_s,El,nq, &
      GS,Sigma,nwork,work)
    use intrinsic_missing, only : EYE
! ********************
! * INPUT variables  *
! ********************
    complex(dp), intent(in) :: ZEnergy
    integer,  intent(in) :: no_u, no_s
    type(Elec), intent(in) :: El
    integer,  intent(in) :: nq
    complex(dp), dimension(no_u,no_u,nq), intent(inout) :: GS
! ********************
! * OUTPUT variables *
! ********************
    complex(dp), intent(inout) :: Sigma(no_s,no_s)

    integer,     intent(in)    :: nwork
    complex(dp), intent(inout) :: work(no_s,no_s,2)
! ********************
! * LOCAL variables  *
! ********************
    integer :: ierr
    integer :: io, jo
    integer :: ipvt(no_s)

#ifndef TS_NOCHECKS
    if ( nwork < no_s ** 2 * 2 ) &
        call die('elec_se-Sigma: worksize too small!')
#endif
    
    call update_UC_expansion(ZEnergy,no_u,no_s,El,nq, &
        El%na_used,El%lasto_used,El%HA,El%SA,GS,nwork, &
        work(1,1,2), Sigma(1,1))

    if ( nq == 1 ) then
#ifndef TS_NOCHECKS
      if ( no_u /= no_s ) call die('no_E/=no_s')
#endif

      ! When no repetition we save it "as is"
      call zcopy(no_s*no_s,GS(1,1,1),1,Sigma(1,1),1)
      
    else if ( El%no_u /= El%no_used ) then

      ! Invert Sigma only when the electrode size is
      ! reduced, and not pre-expanded

      call zgetrf(no_s, no_s, Sigma, no_s, ipvt, ierr )
      if ( ierr /= 0 ) &
          write(*,'(a,i0)') &
          'Inversion of surface Green (A) function failed: ',ierr
      call zgetri(no_s, Sigma, no_s, ipvt, work(1,1,1), no_s**2, ierr)
      if ( ierr /= 0 ) &
          write(*,'(a,i0)') &
          'Inversion of surface Green (B) function failed: ',ierr

    end if
    
    ! Do:
    ! \Sigma = Z*S - H - \Sigma_bulk
!$OMP parallel do default(shared), private(io,jo), collapse(2)
    do jo = 1 , no_s
      do io = 1 , no_s
        Sigma(io,jo) = work(io,jo,2) - Sigma(io,jo)
      end do
    end do
!$OMP end parallel do

  end subroutine UC_expansion_Sigma

  subroutine UC_expansion_Sigma_GammaT(ZEnergy,no_u,no_s,El,nq, &
      GS,Sigma,GammaT,nwork,work)
    use intrinsic_missing, only: EYE
! ********************
! * INPUT variables  *
! ********************
    complex(dp), intent(in) :: ZEnergy
    integer,  intent(in) :: no_u, no_s
    type(Elec), intent(in) :: El
    integer,  intent(in) :: nq
    complex(dp), dimension(no_u,no_u,nq), intent(inout) :: GS
! ********************
! * OUTPUT variables *
! ********************
    complex(dp), intent(inout) :: Sigma(no_s,no_s)
    complex(dp), intent(inout) :: GammaT(no_s,no_s)

    integer,  intent(in) :: nwork
    complex(dp), intent(inout) :: work(no_s,no_s,2)
! ********************
! * LOCAL variables  *
! ********************
    complex(dp), parameter :: zi = dcmplx(0._dp,1._dp)
    integer :: ierr
    integer :: io,jo
    integer :: ipvt(no_s)
    integer, pointer :: p_G(:)

#ifndef TS_NOCHECKS
    if ( nwork < no_s ** 2 * 2 ) &
        call die('elec_se-Sigma-GT: worksize too small. Error')
#endif

#ifdef TBTRANS
    call die('elec_se: GT: This routine should never be called in &
        &TBtrans. Will produce erroneous results.')
#endif

    call update_UC_expansion(ZEnergy,no_u,no_s,El,nq, &
        El%na_used,El%lasto_used,El%HA,El%SA,GS,nwork, &
        work(1,1,2), Sigma(1,1))

    if ( nq == 1 ) then
#ifndef TS_NOCHECKS
      if ( no_u /= no_s ) call die('no_E/=no_s')
#endif

      ! When no repetition we save it "as is"
      call zcopy(no_s*no_s,GS(1,1,1),1,Sigma(1,1),1)
      
    else if ( El%no_u /= El%no_used ) then

      ! Invert Sigma only when the electrode size is
      ! reduced, and not pre-expanded

      call zgetrf(no_s, no_s, Sigma, no_s, ipvt, ierr )
      if ( ierr /= 0 ) &
          write(*,'(a,i0)') &
          'Inversion of surface Green (A) function failed (G): ',ierr
      call zgetri(no_s, Sigma, no_s, ipvt, work(1,1,1), no_s**2, ierr)
      if ( ierr /= 0 ) &
          write(*,'(a,i0)') &
          'Inversion of surface Green (B) function failed (G): ',ierr

    end if

    ! Get pivoting table for the scattering matrix
    ! Note that we here pivot directly into the
    ! the same order of the Green function
    ! to not do it "twice"
    p_G => El%inDpvt%r

!$OMP parallel default(shared), private(io,jo)

    if ( El%Bulk ) then

      ! Do:
      ! work = Z*S - H - (Z*S - H - \Sigma_bulk)
!$OMP do collapse(2)
      do jo = 1 , no_s
        do io = 1 , no_s
          work(io,jo,2) = work(io,jo,2) - Sigma(io,jo)
        end do
      end do
!$OMP end do
      
      ! Do (i.e. store the transposed Gamma)
      ! \Gamma ^ T = i (\Sigma - \Sigma^\dagger)^T
      if ( associated(p_G) ) then
!$OMP do
        do jo = 1 , no_s
          do io = 1 , jo - 1
            GammaT(jo,io) = zi * (work(p_G(io),p_G(jo),2) &
                - dconjg(work(p_G(jo),p_G(io),2)))
            GammaT(io,jo) = zi * (work(p_G(jo),p_G(io),2) &
                - dconjg(work(p_G(io),p_G(jo),2)))
          end do
          io = p_G(jo)
          GammaT(jo,jo) = zi * (work(io,io,2)-dconjg(work(io,io,2)))
        end do
!$OMP end do nowait
      else ! no pivoting
!$OMP do
        do jo = 1 , no_s
          do io = 1 , jo - 1
            GammaT(jo,io) = zi * (work(io,jo,2) &
                - dconjg(work(jo,io,2)))
            GammaT(io,jo) = zi * (work(jo,io,2) &
                - dconjg(work(io,jo,2)))
          end do
          GammaT(jo,jo) = zi * (work(jo,jo,2)-dconjg(work(jo,jo,2)))
        end do
!$OMP end do nowait
      end if
      
    else
       
      ! Do:
      ! \Sigma = Z*S - H - (Z*S - H - \Sigma_bulk)
!$OMP do collapse(2)
      do jo = 1 , no_s
        do io = 1 , no_s
          Sigma(io,jo) = work(io,jo,2) - Sigma(io,jo)
        end do
      end do
!$OMP end do 

      ! Do (i.e. store the transposed Gamma)
      ! \Gamma ^ T = i (\Sigma - \Sigma^\dagger)^T
      if ( associated(p_G) ) then
!$OMP do 
        do jo = 1 , no_s
          do io = 1 , jo - 1
            GammaT(jo,io) = zi * (Sigma(p_G(io),p_G(jo)) &
                - dconjg(Sigma(p_G(jo),p_G(io))))
            GammaT(io,jo) = zi * (Sigma(p_G(jo),p_G(io)) &
                - dconjg(Sigma(p_G(io),p_G(jo))))
          end do
          io = p_G(jo)
          GammaT(jo,jo) = zi * (Sigma(io,io)-dconjg(Sigma(io,io)))
        end do
!$OMP end do nowait

      else ! no pivoting
        
!$OMP do 
        do jo = 1 , no_s
          do io = 1 , jo - 1
            GammaT(jo,io) = zi * (Sigma(io,jo) &
                - dconjg(Sigma(jo,io)))
            GammaT(io,jo) = zi * (Sigma(jo,io) &
                - dconjg(Sigma(io,jo)))
          end do
          GammaT(jo,jo) = zi * (Sigma(jo,jo)-dconjg(Sigma(jo,jo)))
        end do
!$OMP end do nowait

      end if
      
    end if

!$OMP end parallel 

  end subroutine UC_expansion_Sigma_GammaT

  subroutine update_UC_expansion(ZEnergy,no_u,no_s,El,nq,&
       na_u,lasto,H,S,GS,nwork,HSE,GSE)
    use units, only : Pi
! ********************
! * INPUT variables  *
! ********************
    complex(dp), intent(in) :: ZEnergy
    integer,  intent(in) :: no_u, no_s
    type(Elec), intent(in) :: El
    integer,  intent(in) :: nq, na_u,lasto(0:na_u)
    complex(dp), dimension(no_u,no_u,nq), intent(in) :: H, S, GS
! ********************
! * OUTPUT variables *
! ********************
    integer,     intent(in)    :: nwork
    complex(dp), intent(inout) :: HSE(no_s,no_s), GSE(no_s,no_s)
! ********************
! * LOCAL variables  *
! ********************
    integer :: iuo, juo, iow

    if ( nq == 1 ) then
#ifndef TS_NOCHECKS
      if ( no_u /= no_s ) call die('no_E/=no_s')
#endif

      ! In case the pre-expansion is not done on H, S
      if ( El%pre_expand == 1 .and. product(El%Bloch) > 1 ) then

        ! Note that this is because the interface for H and S
        iow = El%no_used
!$OMP parallel do default(shared), private(iuo,juo), collapse(2)
        do juo = 1 , iow
          do iuo = 1 , no_s
            GSE(iuo,juo) = ZEnergy * S(iuo,juo,1) - H(iuo,juo,1)
          end do
        end do
!$OMP end parallel do
        
        call update_UC_expansion_A(iow,no_s,El,nq,na_u,lasto,&
            GSE(1,1),HSE(1,1))

      else

        ! We do not need to copy over GS, as it is
        ! used correctly
        
!$OMP parallel do default(shared), private(iuo,juo), collapse(2)
        do juo = 1 , no_s
          do iuo = 1 , no_s
            !GSE(iuo,juo,1) = GS(iuo,juo,1)
            HSE(iuo,juo) = ZEnergy * S(iuo,juo,1) - H(iuo,juo,1)
          end do
        end do
!$OMP end parallel do

      end if

    else if ( El%repeat ) then
      call repeat(H, S, GS)
    else
      call tile(H, S, GS)
    end if

  contains

    subroutine repeat(H, S, GS)
      complex(dp), dimension(no_u,no_u,El%Bloch(1),El%Bloch(2),El%Bloch(3)), intent(in) :: H, S, GS
      integer :: B(3), i1, i2, i3
      integer :: iau,ia1,ia2,ia3
      integer :: jow,jau,ja1,ja2,ja3
      complex(dp) :: p(3), pZ, qPi
      real(dp) :: rPi(3), wq
      
!$OMP parallel default(shared), private(wq,rPi,qPi,p,pZ), &
!$OMP&  private(B,i1,i2,i3), &
!$OMP&  private(iow,iau,ia1,ia2,ia3,iuo), &
!$OMP&  private(jow,jau,ja1,ja2,ja3,juo)

!$OMP workshare
      GSE(:,:) = 0._dp
      HSE(:,:) = 0._dp
!$OMP end workshare

      ! Save some multiplications
      B(:) = El%Bloch(:)
      wq = log(1._dp / real(nq,dp))

      do i3 = 1, B(3)
      do i2 = 1, B(2)
      do i1 = 1, B(1)

       rPi = 2._dp * Pi * (q_exp(El,i1,i2,i3) + El%bkpt_cur)
       qPi = cdexp(dcmplx(0._dp,rPi(1)))

!$OMP do collapse(4)
       do iau = 1 , na_u
        do ia3 = 1 , B(3)
        do ia2 = 1 , B(2)
        do ia1 = 1 , B(1)

          p(3) = cdexp(dcmplx(wq,-ia1*rPi(1)-ia2*rPi(2)-ia3*rPi(3)))
          iow = lasto(iau-1) * nq + &
              (( (ia3-1)*B(2) + (ia2-1) ) * B(1) + (ia1-1)) * (lasto(iau) - lasto(iau-1))

          do iuo = 1 + lasto(iau-1) , lasto(iau)
           iow = iow + 1
            
           jow = 0
           do jau = 1 , na_u
            do ja3 = 1 , B(3)
            p(2) = p(3)*cdexp(dcmplx(0._dp,ja3*rPi(3)))
            do ja2 = 1 , B(2)
            p(1) = p(2)*cdexp(dcmplx(0._dp,ja2*rPi(2)))
            do ja1 = 1 , B(1)
              ! This takes one additional phase per iteration
              p(1) = p(1) * qPi
              pZ = p(1) * ZEnergy
              do juo = 1 + lasto(jau-1) , lasto(jau)
               jow = jow + 1
                
               GSE(jow,iow) = GSE(jow,iow) + p(1) * GS(juo,iuo,i1,i2,i3)
               HSE(jow,iow) = HSE(jow,iow) + pZ * S(juo,iuo,i1,i2,i3) - p(1) * H(juo,iuo,i1,i2,i3)
                
              end do !juo
            end do !ja1
            end do !ja2
            end do !ja3
           end do !jau
          end do !iuo
        end do !ia1
        end do !ia2
        end do !ia3
       end do !iau
!$OMP end do

      end do !i1
      end do !i2
      end do !i3

!$OMP end parallel

    end subroutine repeat

    subroutine tile(H, S, GS)
      complex(dp), dimension(no_u,no_u,El%Bloch(1),El%Bloch(2),El%Bloch(3)), intent(in) :: H, S, GS
      integer :: B(3), i1, i2, i3
      integer :: iau,ia1,ia2,ia3
      integer :: jow,jau,ja1,ja2,ja3
      complex(dp) :: p(3), pZ, qPi
      real(dp) :: rPi(3), wq

!$OMP parallel default(shared), private(wq,rPi,qPi,p,pZ), &
!$OMP&  private(B,i1,i2,i3), &
!$OMP&  private(iow,iau,ia1,ia2,ia3,iuo), &
!$OMP&  private(jow,jau,ja1,ja2,ja3,juo)

!$OMP workshare
      GSE(:,:) = 0._dp
      HSE(:,:) = 0._dp
!$OMP end workshare

      ! Save some multiplications
      B(:) = El%Bloch(:)
      wq = log(1._dp / real(nq,dp))

      do i3 = 1, B(3)
      do i2 = 1, B(2)
      do i1 = 1, B(1)

       rPi = 2._dp * Pi * (q_exp(El,i1,i2,i3) + El%bkpt_cur)
       qPi = cdexp(dcmplx(0._dp,rPi(1)))

!$OMP do collapse(3)
       do ia3 = 1 , B(3)
       do ia2 = 1 , B(2)
       do ia1 = 1 , B(1)

        p(3) = cdexp(dcmplx(wq,-ia1*rPi(1)-ia2*rPi(2)-ia3*rPi(3)))
        iow = (( (ia3-1)*B(2) + (ia2-1) ) * B(1) + (ia1-1)) * no_u
        do iuo = 1, no_u
         iow = iow + 1

         jow = 0
         do ja3 = 1 , B(3)
         p(2) = p(3)*cdexp(dcmplx(0._dp,ja3*rPi(3)))
         do ja2 = 1 , B(2)
         p(1) = p(2)*cdexp(dcmplx(0._dp,ja2*rPi(2)))
         do ja1 = 1 , B(1)
         ! This takes one additional phase per iteration
         p(1) = p(1)*qPi
         pZ = p(1) * ZEnergy
         do juo = 1, no_u
          jow = jow + 1
                
          GSE(jow,iow) = GSE(jow,iow) + p(1) * GS(juo,iuo,i1,i2,i3)
          HSE(jow,iow) = HSE(jow,iow) + pZ * S(juo,iuo,i1,i2,i3) - p(1) * H(juo,iuo,i1,i2,i3)
                
         end do !juo
         end do !ja1
         end do !ja2
         end do !ja3
        end do !iuo
       end do !ia1
       end do !ia2
       end do !ia3
!$OMP end do

      end do !i1
      end do !i2
      end do !i3

!$OMP end parallel

    end subroutine tile
    
  end subroutine update_UC_expansion

  subroutine update_UC_expansion_A(no_u,no_s,El,nq, &
      na_u,lasto,A,AE)
    use units, only : Pi
! ********************
! * INPUT variables  *
! ********************
    integer,  intent(in) :: no_u, no_s
    type(Elec), intent(in) :: El
    integer,  intent(in) :: nq, na_u, lasto(0:na_u)
    complex(dp), intent(in) :: A(no_u,no_u,El%Bloch(1),El%Bloch(2),El%Bloch(3))
! ********************
! * OUTPUT variables *
! ********************
    complex(dp), intent(inout) :: AE(no_s,no_s)

    integer :: B(3), i1, i2, i3
    integer :: iau,iow,ia1,ia2,ia3,iuo
    integer :: jau,jow,ja1,ja2,ja3,juo
    complex(dp) :: p(3), qPi
    real(dp) :: rPi(3), wq
    
    if ( no_u == no_s ) then
      call die('update_UC_expansion_A: error!')
    else if ( El%repeat ) then
      call repeat()
    else
      call tile()
    end if
    
  contains
    
    ! This is the crucial calcuation.
    ! If we use bulk values in the electrodes
    ! we need not add the expanded H and S values to get the 
    ! electrode \Sigma. Hence, we need only expand
    ! surface Green function
    
    subroutine repeat()
      
!$OMP parallel default(shared), private(wq,rPi,qPi,p), &
!$OMP&  private(B,i1,i2,i3), &
!$OMP&  private(iow,iau,ia1,ia2,ia3,iuo), &
!$OMP&  private(jow,jau,ja1,ja2,ja3,juo)

!$OMP workshare
      AE(:,:) = 0._dp
!$OMP end workshare

      ! Save some multiplications
      B(:) = El%Bloch(:)
      wq = log(1._dp / real(nq,dp))

      do i3 = 1, B(3)
      do i2 = 1, B(2)
      do i1 = 1, B(1)

       rPi(:) = 2._dp * Pi * (q_exp(El,i1,i2,i3) + El%bkpt_cur)
       qPi = cdexp(dcmplx(0._dp,rPi(1)))

!$OMP do collapse(4)
       do iau = 1 , na_u
        do ia3 = 1 , B(3)
        do ia2 = 1 , B(2)
        do ia1 = 1 , B(1)
            
          p(3) = cdexp(dcmplx(wq,-ia1*rPi(1)-ia2*rPi(2)-ia3*rPi(3)))
          iow = lasto(iau-1) * nq + &
              (( (ia3-1)*B(2) + (ia2-1) ) * B(1) + (ia1-1)) * (lasto(iau) - lasto(iau-1))
          do iuo = 1 + lasto(iau-1) , lasto(iau)
           iow = iow + 1
           
           jow = 0
           do jau = 1 , na_u
            do ja3 = 1 , B(3)
            p(2) = p(3)*cdexp(dcmplx(0._dp,ja3*rPi(3)))
            do ja2 = 1 , B(2)
            p(1) = p(2)*cdexp(dcmplx(0._dp,ja2*rPi(2)))
            do ja1 = 1 , B(1)
              ! This takes one additional phase per iteration
              p(1) = p(1)*qPi
              do juo = 1 + lasto(jau-1) , lasto(jau)
               jow = jow + 1
               
               AE(jow,iow) = AE(jow,iow) + p(1) * A(juo,iuo,i1,i2,i3)
               
             end do !juo
            end do !ja1
            end do !ja2
            end do !ja3
           end do !jau
          end do !iuo
        end do !ia1
        end do !ia2
        end do !ia3
       end do !iau
!$OMP end do

      end do !i1
      end do !i2
      end do !i3

!$OMP end parallel

    end subroutine repeat
   
    subroutine tile()
          
!$OMP parallel default(shared), private(wq,rPi,qPi,p), &
!$OMP&  private(B,i1,i2,i3), &
!$OMP&  private(iow,iau,ia1,ia2,ia3,iuo), &
!$OMP&  private(jow,jau,ja1,ja2,ja3,juo)

!$OMP workshare
      AE(:,:) = 0._dp
!$OMP end workshare

      ! Save some multiplications
      B(:) = El%Bloch(:)
      wq = log(1._dp / real(nq,dp))

      do i3 = 1, B(3)
      do i2 = 1, B(2)
      do i1 = 1, B(1)

       rPi(:) = 2._dp * Pi * (q_exp(El,i1,i2,i3) + El%bkpt_cur)
       qPi = cdexp(dcmplx(0._dp,rPi(1)))

!$OMP do collapse(3)
       do ia3 = 1 , B(3)
       do ia2 = 1 , B(2)
       do ia1 = 1 , B(1)
            
        p(3) = cdexp(dcmplx(wq,-ia1*rPi(1)-ia2*rPi(2)-ia3*rPi(3)))
        iow = (( (ia3-1)*B(2) + (ia2-1) ) * B(1) + (ia1-1)) * no_u
        do iuo = 1, no_u
         iow = iow + 1
           
         jow = 0
         do ja3 = 1 , B(3)
         p(2) = p(3)*cdexp(dcmplx(0._dp,ja3*rPi(3)))
         do ja2 = 1 , B(2)
         p(1) = p(2)*cdexp(dcmplx(0._dp,ja2*rPi(2)))
         do ja1 = 1 , B(1)
         p(1) = p(1)*qPi
          do juo = 1, no_u
           jow = jow + 1
               
           AE(jow,iow) = AE(jow,iow) + p(1) * A(juo,iuo,i1,i2,i3)
               
          end do !juo
         end do !ja1
         end do !ja2
         end do !ja3
        end do !iuo
       end do !ia1
       end do !ia2
       end do !ia3
!$OMP end do

      end do !i1
      end do !i2
      end do !i3

!$OMP end parallel

    end subroutine tile
   
  end subroutine update_UC_expansion_A

end module m_ts_elec_se
