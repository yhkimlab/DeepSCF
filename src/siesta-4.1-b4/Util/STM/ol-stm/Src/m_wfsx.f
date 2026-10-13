! ---
! Copyright (C) 1996-2016       The SIESTA group
!  This file is distributed under the terms of the
!  GNU General Public License: see COPYING in the top directory
!  or http://www.gnu.org/copyleft/gpl.txt .
! See Docs/Contributors.txt for a list of contributors.
! ---
      module m_wfsx

      IMPLICIT NONE

      private

      integer, parameter :: dp = kind(1.d0)
      integer, parameter :: sp = selected_real_kind(6,20)

      public :: wfsx_get_info, read_wfsx

      CONTAINS
      
      SUBROUTINE wfsx_get_info(unit,NSPIN,NORB,NUMWF,NK,gamma)
      INTEGER, intent(in) :: unit, NSPIN, NORB
      integer, intent(out):: NUMWF, NK
      logical, intent(out) :: gamma
      
      integer :: nsp, nuo, ik, numberwf, iwf
      
        READ(UNIT) NK, gamma
        READ(UNIT) NSP
        IF (NSP .NE. NSPIN) THEN
          WRITE(6,*) 'NSPIN is not consistent between data files!'
          STOP
        ENDIF
        READ(UNIT) NUO
        IF (NUO .NE. NORB) THEN
          WRITE(6,*) 'Nr. of orbs is not consistent between data files!'
          STOP
        ENDIF

        READ(UNIT)              ! skip over orbital info

        NUMWF = 0
        DO IK = 1, NK
          ! will elide inner loop over spin, as we only want
          ! basic info 
          READ(UNIT) ! skip over k-point coords
          READ(UNIT) ! skip over spin index
          READ(UNIT) NUMBERWF
          IF (NUMBERWF .GT. NUMWF) NUMWF = NUMBERWF
          DO IWF = 1, NUMBERWF
            READ(UNIT)  ! skip over wf index
            READ(UNIT)  ! skip over energy
            READ(UNIT)  ! skip over wf data
          ENDDO
        ENDDO
        end subroutine wfsx_get_info
      
      SUBROUTINE read_wfsx(unit,NSPIN,NORB,NWF,NUMWF,NK,
     .                     RPSI,IPSI,E,K,IND)

      INTEGER, intent(in) :: unit, NSPIN, NORB
      integer, intent(in) :: NUMWF, NK
      integer, intent(out) :: NWF(NK), IND(NK,NUMWF)
      real(dp),intent(out) :: RPSI(NORB,NK,NUMWF,NSPIN), 
     .     IPSI(NORB,NK,NUMWF,NSPIN),
     .     E(NK,NUMWF,NSPIN), K(NK,3)
      
C Reads the wavefunctions and energies from a file written by Siesta
C P. Ordejon, July 2003
C Modified for complex wavefunctions and multiple k-points,
C P. Ordejon, July 2004

!   WFSX version: Alberto Garcia, February 2017
      

! INTEGER NSPIN     : Number of spin components
! INTEGER NORB      : Number of basis orbitals

! NK        : Number of k-points to read 
! NUMWF     : Max num of wavefncts to read for a given k-po.
! NWF(NK)   : Number of wavefunctions to read for each k-po.
! IND(NK,NUMWF)           : List of indexes of wavefunctions
      
! RPSI(NORB,NK,NUMWF,NSPIN): Wavefunctions (real part)
! IPSI(NORB,NK,NUMWF,NSPIN): Wavefunctions (imag part)
! E(NK,NUMWF,NSPIN)        : Eigenvalues
! K(NK,3)                  : K-points

      INTEGER NSP, NUO, ISPIN, IISPIN, IWF, IORB
      INTEGER NUMK, IK, IIK

      real(sp), allocatable :: rsp(:), isp(:)
      
      logical :: gamma

        READ(UNIT) NUMK, gamma
        IF (NK .NE. NUMK) THEN
          WRITE(6,*) 'Error in number of k-points'
          STOP
        ENDIF
        READ(UNIT) NSP
        IF (NSP .NE. NSPIN) THEN
          WRITE(6,*) 'NSPIN is not consistent between data files!'
          STOP
        ENDIF
        READ(UNIT) NUO
        IF (NUO .NE. NORB) THEN
          WRITE(6,*) 'Nr. of orbs is not consistent between data files!'
          STOP
        ENDIF

        read(unit)              ! Skip over orbital info
        
        if (gamma) then
           allocate(rsp(norb))
        else
           allocate(rsp(norb),isp(norb))
        endif
        
        DO IK = 1, NK
          DO ISPIN = 1,NSPIN
            READ(UNIT) IIK,K(IK,1),K(IK,2),K(IK,3)
            IF (IK .NE. IIK) THEN
              WRITE(6,*) 'Inconsistent order of kpoints in WFSX file!'
              STOP
            ENDIF
            READ(UNIT) IISPIN
            IF (IISPIN .NE. ISPIN) THEN
              WRITE(6,*) 'Inconsistent order of spins in WFSX file!'
              STOP
            ENDIF
            READ(UNIT) NWF(IK)

            DO IWF = 1,NWF(IK)
              READ(UNIT) IND(IK,IWF)
              READ(UNIT) E(IK,IWF,ISPIN)
              if (gamma) then
                 READ(UNIT) ( rsp(iorb), iorb=1,norb)
                 RPSI(1:norb,IK,IWF,ISPIN) = real(rsp(1:norb),kind=dp)
                 IPSI(1:norb,IK,IWF,ISPIN) = 0.0_dp
              else
                 READ(UNIT) ( rsp(iorb), isp(iorb), iorb=1,norb)
                 RPSI(1:norb,IK,IWF,ISPIN) = real(rsp(1:norb),kind=dp)
                 IPSI(1:norb,IK,IWF,ISPIN) = real(isp(1:norb),kind=dp)
              endif
            ENDDO
          ENDDO
        ENDDO

        if (gamma) then
           deallocate(rsp)
        else
           deallocate(rsp,isp)
        endif

        END subroutine read_wfsx
      end module m_wfsx
        

