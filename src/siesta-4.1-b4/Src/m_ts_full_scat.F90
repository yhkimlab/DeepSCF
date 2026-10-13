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

module m_ts_full_scat

  use precision, only : dp

  use m_ts_electype
  use m_ts_cctype

  implicit none

  private

  public :: calc_GF
  public :: calc_GF_Bias
  public :: calc_GF_Part
  public :: GF_Gamma_GF
  public :: insert_Self_Energies

contains

  
! Full converted GF.G.GF^\dagger routine for speed.
! This routine is extremely fast compared to any previous implementation.
! It relies on the fact that Gf only contains the electrode columns.
  subroutine GF_Gamma_GF(El, no_u_TS, no, GF, &
       GGG,nwork,work)

!  This routine returns GGG=GF.Gamma.GF^\dagger, where GF is a (no_u)x(no)
!  matrix and the states
!  corresponds to the (no) electrode states
!  Gamma is a (no)x(no) matrix.

! *********************
! * INPUT variables   *
! *********************
    ! electrode self-energy
    type(Elec), intent(in) :: El
    integer, intent(in) :: no_u_TS ! no. states in contact region
    integer, intent(in) :: no      ! no. states for all electrodes
    ! The Green function (it has to be the column that corresponds to the electrode)
    complex(dp), intent(inout) :: GF(no_u_TS,no)
    ! A work array for doing the calculation... (nwork has to be larger than no_u_TS)
    integer,     intent(in)    :: nwork
    complex(dp), intent(inout) :: work(nwork)

! *********************
! * OUTPUT variables  *
! *********************
    complex(dp), intent(out) :: GGG(no_u_TS*no_u_TS)    !GF.Gamma.GF^\dagger

! *********************
! * LOCAL variables   *
! *********************
    complex(dp), parameter :: z0 = dcmplx(0._dp, 0._dp)
    complex(dp), parameter :: z1 = dcmplx(1._dp, 0._dp)
    complex(dp), parameter :: zi = dcmplx(0._dp, 1._dp)

    integer :: i, NB, ind, iB

#ifdef TRANSIESTA_DEBUG
    call write_debug( 'PRE GFGammaGF' )
#endif

    call timer("GFGGF",1)

    ! Number of times we can divide the large matrix
    NB = no_u_TS / no
       
    ! Loop over bottom row matrix 
    do iB = 0 , NB - 1
       
       ! Collect the top row of complex conjugated Gf
       ind = no_u_TS * no * iB + 1
       do i = 1 , no
          GGG(ind:ind-1+no) = dconjg(Gf(iB*no+1:(iB+1)*no,i))
          ind = ind + no
       end do
       ind = no_u_TS * no * iB + 1
       
       ! Do Gamma.Gf^\dagger
#ifdef USE_GEMM3M
       call zgemm3m( &
#else
       call zgemm( &
#endif
            'T','T',no,no,no,z1, &
            El%Gamma, no, &
            GGG(ind), no, &
            z0, work,no)
       
       ! Calculate the Gf.Gamma.Gf^\dagger product for the entire column
#ifdef USE_GEMM3M
       call zgemm3m( &
#else
       call zgemm( &
#endif
            'N','N',no_u_TS,no,no,z1, &
            Gf(1,1), no_u_TS, &
            work   ,      no, &
            z0, GGG(ind),no_u_TS)
    
    end do

    ! in case the block size does not match the matrix order
    if ( NB * no /= no_u_TS ) then

       ! The size of the remaining block
       iB = no_u_TS - NB * no

       ! Copy over the block
       ind = no_u_TS * no * NB + 1
       do i = 1 , no
          ! So this is the complex conjugated of the iB'th block
          GGG(ind:ind-1+iB) = dconjg(Gf(NB*no+1:NB*no+iB,i))
          ind = ind + iB
       end do
       ind = no_u_TS * no * NB + 1

       ! Do Gamma.Gf^\dagger
#ifdef USE_GEMM3M
       call zgemm3m( &
#else
       call zgemm( &
#endif
            'T','T',no,iB,no,z1, &
            El%Gamma, no, &
            GGG(ind), iB, &
            z0, work,no)
       
       ! Calculate the Gf.Gamma.Gf^\dagger product for the entire column
#ifdef USE_GEMM3M
       call zgemm3m( &
#else
       call zgemm( &
#endif
            'N','N',no_u_TS,iB,no,z1, &
            Gf(1,1), no_u_TS, &
            work   ,      no, &
            z0, GGG(ind),no_u_TS)

    end if


#ifdef TRANSIESTA_31
    ! Lets try and impose symmetry...
    call my_symmetrize(no_u_TS,GGG)
#endif

    call timer("GFGGF",2)

#ifdef TRANSIESTA_DEBUG
    call write_debug( 'POS GFGammaGF' )
#endif

  end subroutine GF_Gamma_GF

#ifdef TRANSIESTA_31
subroutine my_symmetrize(N,M)
  use parallel, only : IONode
    integer    , intent(in) :: N
    complex(dp), intent(inout) :: M(N,N)
    integer :: i,j
    do j = 1 , N
       do i = 1 , j
!          if(ionode)print *,M(j,i),M(i,j)
          M(j,i) = dimag(M(i,j))
          M(i,j) = M(j,i)

       end do
    end do
  end subroutine my_symmetrize
#endif

! ##################################################################
! ## Calculating Full Green functions of                          ## 
! ##                                                              ##          
! ##                            By                                ##
! ##              Mads Brandbyge, mbr@mic.dtu.dk                  ##
! ##                                                              ##
! ##                                                              ##
! ## Completely restructured to be able to handle sparse matrices ##
! ##                                                              ##
! ##                                                              ##
! ##  Modified by Nick Papior Andersen                            ##
! ##################################################################
  subroutine calc_GF(cE,no_u_TS,GFinv,GF)
    
    use intrinsic_missing, only: EYE
    use precision, only: dp

    implicit none 

! *********************
! * INPUT variables   *
! *********************
    type(ts_c_idx), intent(in) :: cE
    ! Sizes of the different regions...
    integer, intent(in) :: no_u_TS
    ! Work should already contain Z*S - H
    ! This may seem strange, however, it will clean up this routine extensively
    ! as we dont need to make two different routines for real and complex
    ! Hamiltonian values.
    complex(dp), intent(in out) :: GFinv(no_u_TS,no_u_TS) ! the inverted GF
    complex(dp), intent(out) :: GF(no_u_TS**2)

! Local variables
    integer :: ipvt(no_u_TS), ierr

    if ( cE%fake ) return

#ifdef TRANSIESTA_DEBUG
    call write_debug( 'PRE getGF' )
#endif

    call timer('GFT',1) 

    call EYE(no_u_TS,GF)
    
    ! Invert directly
    call zgesv(no_u_TS,no_u_TS,GFinv,no_u_TS,ipvt,GF,no_u_TS,ierr)            
    if ( ierr /= 0 ) call die('GF: Could not invert the Green function')
       
    call timer('GFT',2)  

#ifdef TRANSIESTA_DEBUG
    call write_debug( 'POS getGF' )
#endif

  end subroutine calc_GF


! ##################################################################
! ## Calculating Green functions in the reigon of the electrodes  ## 
! ##                                                              ##          
! ##  Fully created by Nick Papior Andersen, nickpapior@gmail.com ##
! ##################################################################
  subroutine calc_GF_Bias(cE,no_u_TS,N_Elec,Elecs,GFinv,GF)
    
    use precision, only: dp

    use m_ts_method, only : orb_offset

    implicit none 

! *********************
! * INPUT variables   *
! *********************
    type(ts_c_idx), intent(in) :: cE
    ! Sizes of the different regions...
    integer, intent(in) :: no_u_TS
    ! Electrodes
    integer, intent(in) :: N_Elec
    type(Elec), intent(in) :: Elecs(N_Elec)
    ! Work should already contain Z*S - H
    ! This may seem strange, however, it will clean up this routine extensively
    ! as we dont need to make two different routines for real and complex
    ! Hamiltonian values.
    complex(dp), intent(in out) :: GFinv(no_u_TS,no_u_TS) ! the inverted GF
    ! We only need Gf in the left and right blocks...
    complex(dp), intent(out) :: GF(:)

! Local variables
    integer :: ipvt(no_u_TS)
    integer :: i, o, no, iEl, off_row

    if ( cE%fake ) return

#ifdef TRANSIESTA_DEBUG
    call write_debug( 'PRE getGF' )
#endif

    call timer('GFTB',1) 

    no = sum(TotUsedOrbs(Elecs(:)))
    if ( no == 0 ) call die('GFB: Error in contour setup')
    if ( no * no_u_TS > size(GF) ) &
         call die('GFB: Wrong size of Green function')

    ! Create the RHS for inversion...
    GF(:) = dcmplx(0._dp,0._dp)

    o = 0
    do iEl = 1 , N_Elec
       i = Elecs(iEl)%idx_o
       off_row = i - orb_offset(i) - 1
       do i = 1 , TotUsedOrbs(Elecs(iEl))
          GF(o*no_u_TS+off_row+i) = dcmplx(1._dp,0._dp)
          o = o + 1
       end do
    end do

    ! Invert directly
    call zgesv(no_u_TS,o,GFinv,no_u_TS,ipvt,GF,no_u_TS,i)
    if ( i /= 0 ) call die('GFB: Could not invert the Green function')
       
    call timer('GFTB',2)  

#ifdef TRANSIESTA_DEBUG
    call write_debug( 'POS getGF' )
#endif

  end subroutine calc_GF_Bias


! ##################################################################
! ## Calculating Full Green functions of                          ## 
! ##                                                              ##          
! ##                            By                                ##
! ##              Mads Brandbyge, mbr@mic.dtu.dk                  ##
! ##                                                              ##
! ##                                                              ##
! ## Completely restructured to be able to handle sparse matrices ##
! ##                                                              ##
! ##                                                              ##
! ##  Modified by Nick Papior Andersen                            ##
! ##################################################################
  subroutine calc_GF_Part(cE,no_u_TS, N_Elec, Elecs, & ! Size of the problem
       GFinv,GF)
    
    use intrinsic_missing, only: EYE
    use precision, only: dp
    use m_ts_method, only : orb_offset

    implicit none 

! *********************
! * INPUT variables   *
! *********************
    type(ts_c_idx), intent(in) :: cE
    ! Sizes of the different regions...
    integer, intent(in) :: no_u_TS
    integer, intent(in) :: N_Elec
    type(Elec), intent(in) :: Elecs(N_Elec)
    ! Work should already contain Z*S - H
    ! This may seem strange, however, it will clean up this routine extensively
    ! as we dont need to make two different routines for real and complex
    ! Hamiltonian values.
    complex(dp), intent(in out) :: GFinv(no_u_TS,no_u_TS) ! the inverted GF
    complex(dp), intent(out) :: GF(:)

! Local variables
    integer :: ipvt(no_u_TS)
    integer :: i, j, no
    logical :: found

    if ( cE%fake ) return

#ifdef TRANSIESTA_DEBUG
    call write_debug( 'PRE getGF' )
#endif

    call timer('GFT_P',1) 

    no = no_u_TS
    do i = 1, N_Elec
       if ( Elecs(i)%DM_update == 0 ) then
          no = no - TotUsedOrbs(Elecs(i))
       end if
    end do
    if ( no * no_u_TS > size(GF) ) &
         call die('GFP: Wrong size of Green function')

    ! initialize
    GF(:) = dcmplx(0._dp,0._dp)

    i = 0
    do j = 0 , no - 1
       found = .false.
       do while ( .not. found ) 
          i = i + 1
          if ( .not. any(OrbInElec(Elecs,i) .and. Elecs(:)%DM_update == 0) ) then
             
             GF(j*no_u_TS+i-orb_offset(i)) = dcmplx(1._dp,0._dp)
             found = .true.
          end if
       end do
    end do

    ! Invert directly
    call zgesv(no_u_TS,no,GFinv,no_u_TS,ipvt,GF,no_u_TS,i)
    if ( i /= 0 ) call die('GFP: Could not invert the Green function')

    call timer('GFT_P',2)

#ifdef TRANSIESTA_DEBUG
    call write_debug( 'POS getGF' )
#endif

  end subroutine calc_GF_Part

  subroutine insert_Self_Energies(no_u, Gfinv, El)
    use m_ts_method, only : orb_offset
    integer, intent(in) :: no_u
    complex(dp), intent(in out) :: GFinv(no_u,no_u)
    type(Elec), intent(in) :: El
    
    integer :: i, j, ii, jj, iii, off, no
    
    no = TotUsedOrbs(El)
    off = El%idx_o - orb_offset(El%idx_o) - 1
    
    if ( El%Bulk ) then
!$OMP do private(j,jj,i,ii)
       do j = 0 , no - 1
          jj = off + j + 1
          ii = j * no
          do i = 1 , no
             Gfinv(off+i,jj) = El%Sigma(ii+i)
          end do
       end do
!$OMP end do nowait
    else
!$OMP do private(j,jj,i,ii,iii)
       do j = 0 , no - 1
          iii = j * no
          jj = off + j + 1
          do i = 1 , no
             ii  = off + i
             Gfinv(ii,jj) = Gfinv(ii,jj) - El%Sigma(iii+i) 
          end do
       end do
!$OMP end do nowait
    end if

  end subroutine insert_Self_Energies

end module m_ts_full_scat
