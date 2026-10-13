! ---
! Copyright (C) 1996-2016	The SIESTA group
!  This file is distributed under the terms of the
!  GNU General Public License: see COPYING in the top directory
!  or http://www.gnu.org/copyleft/gpl.txt .
! See Docs/Contributors.txt for a list of contributors.
! ---
module m_energies
  use precision, only: dp
  implicit none

  private :: dp
  public
  save
  
  real(dp):: DEharr     ! Tr[H * (DM_out - DM_in)], for Harris energy
  real(dp):: DEna       ! Neutral-atom energy term, calculated  in dnaefs
  real(dp):: DUext      ! Interaction energy with external  electric field,
                        ! calculated in dhscf

  real(dp):: DUscf      ! Electrostatic energy of (rhoscf- rhoatm), calc. in dhscf
  real(dp):: Dxc        ! Integral((epsxc-Vxc)*Rho), calculated  in dhscf (not used)
  real(dp):: Ecorrec    ! Energy term eta*DeltaQ, calculated in  ordern
  real(dp):: ef         ! Fermi energy
  real(dp):: efs(2)     ! Fermi energies (only for fixed spin calculations)
  real(dp):: Eharrs     ! Harris-functional total energy
  real(dp):: Eions      ! Self-energy of isolated ions
  real(dp):: Ekin       ! Kinetic energy of electrons,  calculated in kinefsm
  real(dp):: Ekinion    ! Kinetic energy of ions
  real(dp):: Emad       ! Madelung energy term, calculated in  madelung
  real(dp):: Ena        ! Neutral-atom term in the total energy,  calculated in naefs
  real(dp):: Enaatm     ! Integral of Vna * rhoatm, calculated  in dhscf
  real(dp):: Enascf     ! Integral of Vna * rhoscf, calculated  in dhscf
  real(dp):: Enl        ! Non-local pseudopot. energy term,  calculated in nlefsm
  real(dp):: Emeta      ! Metadynamics energy contribution  calculated in meta
  real(dp):: Entropy    ! Entropy due to electron state occupations
  real(dp):: Etot       ! Total electronic energy
  real(dp):: Exc        ! Exchange-correlation energy,  calculated in dhscf
  real(dp):: E0         ! Non-SCF part of total energy
  real(dp):: Emm        ! Classical two-body term, calculated in  twobody
  real(dp):: FreeE      ! Free energy
  real(dp):: FreeEharris! Free energy computed with Harris total energy
  real(dp):: Uatm       ! Harris hartree electron energy,  calculated in dhscf
  real(dp):: Uscf       ! SCF hartree electron energy,  calculated in dhscf
  real(dp):: Ebs        ! Band-structure energy, Tr(DM*H), calculated in compute_dm
  real(dp):: Eso        ! Spin-orbit energy
  real(dp):: Eldau      
  real(dp):: DEldau

  real(dp) :: DE_NEGF  ! NEGF total energy contribution = - e * \sum_i N_i \mu_i

contains

  !> Initialize ALL energies to 0.
  subroutine init_Energies()

    DEharr = 0._dp
    DEna = 0._dp
    DUext = 0._dp
    DUscf = 0._dp
    Dxc = 0._dp
    Ecorrec = 0._dp
    ef = 0._dp
    efs = 0._dp
    Eharrs = 0._dp
    Eions = 0._dp
    Ekin = 0._dp
    Ekinion = 0._dp
    Emad = 0._dp
    Ena = 0._dp
    Enaatm = 0._dp
    Enascf = 0._dp
    Enl = 0._dp
    Emeta = 0._dp
    Entropy = 0._dp
    Etot = 0._dp
    Exc = 0._dp
    E0 = 0._dp
    Emm = 0._dp
    FreeE = 0._dp
    FreeEharris = 0._dp
    Uatm = 0._dp
    Uscf = 0._dp
    Ebs = 0._dp
    Eso = 0._dp
    Eldau = 0._dp      
    DEldau = 0._dp
    DE_NEGF = 0._dp

  end subroutine init_Energies

  !> To ease the computation of specific deferred
  !> quantites we allow the computation of these quantites
  !> In ONE place

  subroutine update_DEna()

    DEna = Enascf - Enaatm
    
  end subroutine update_DEna

  subroutine update_E0()

    E0 = Ena + Ekin + Enl + Eso - Eions
    
  end subroutine update_E0
  
  subroutine update_Etot()
    
    ! DUext (external electric field) -- should it be in or out?
    Etot = Ena + Ekin + Enl + Eso - Eions + &
         DEna + DUscf + DUext + Exc + &
         Ecorrec + Emad + Emm + Emeta + Eldau
    ! Commented out the NEGF contribution to the total energy
    ! We know it is wrong, but we estimate it. See output
    !    Etot = Etot + DE_NEGF

  end subroutine update_Etot

  !> @param kBT the temperature in energy
  subroutine update_FreeE( kBT )
    real(dp), intent(in) :: kBT

    FreeE = Etot - kBT * Entropy

  end subroutine update_FreeE

  !> @param kBT the temperature in energy
  subroutine update_FreeEHarris( kBT )
    real(dp), intent(in) :: kBT

    FreeEHarris = Eharrs - kBT * Entropy

  end subroutine update_FreeEHarris

end module m_energies



