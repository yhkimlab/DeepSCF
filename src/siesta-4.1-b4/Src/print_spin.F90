subroutine print_spin(qspin)
  !
  ! Prints spin in output and CML files
  !
  use m_spin,          only: nspin
  use atomlist,        only: no_l
  use sparse_matrices, only: listhptr, numh
  use sparse_matrices, only: S, Dscf
  use siesta_cml
  use parallel,        only: IOnode
  use precision,       only: dp
#ifdef MPI
  use m_mpi_utils,     only: globalize_sum
#endif

  implicit none
  
  real(dp), intent(out)  :: qspin(4)

  integer  :: ispin, io, j, ind
  real(dp) :: qaux
  real(dp) :: Stot        ! Total spin magnitude
  real(dp) :: Svec(3)     ! Total spin vector
#ifdef MPI
  real(dp) :: qtmp(4)
#endif

  if (nspin .ge. 2) then
     do ispin = 1,nspin
        qspin(ispin) = 0.0_dp
        do io = 1,no_l
           do j = 1,numh(io)
              ind = listhptr(io)+j
              qspin(ispin) = qspin(ispin) + Dscf(ind,ispin) * S(ind)
           enddo
        enddo
     enddo

#ifdef MPI
     !       Global reduction of spin components
     call globalize_sum(qspin(1:nspin),qtmp(1:nspin))
     qspin(1:nspin) = qtmp(1:nspin)
#endif

     if (nspin .eq. 2) then

        if (IOnode) then
           Svec(1) = 0.0_dp
           Svec(2) = 0.0_dp
           Svec(3) = qspin(1) - qspin(2)
           Stot = Svec(3)
           write(6,'(5x,a,f10.5,2f10.1,f10.5)') 'spin moment: S , {S} = ', Stot, Svec
        endif
        if (cml_p) call cmlAddProperty(xf=mainXML,            &
             value=qspin(1)-qspin(2), dictref='siesta:stot', &
             units='siestaUnits:spin')

     elseif (nspin .eq. 4) then

        call spnvec( nspin, qspin, qaux, Stot, Svec )
        if (IONode) then
           write(6,'(5x,a,4f10.5)') 'spin moment: S , {S} = ', Stot, Svec
           if (cml_p) then
              call cmlAddProperty(xf=mainXML, value=Stot,  &
                   dictref='siesta:stot', units='siestaUnits:spin')
              call cmlAddProperty(xf=mainXML, value=Svec,  &
                   dictref='siesta:svec', units='siestaUnits:spin')
           endif !cml_p
        endif
     endif
  endif
end subroutine print_spin
