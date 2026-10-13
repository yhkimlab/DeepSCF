! 
! Copyright (C) 1996-2016	The SIESTA group
!  This file is distributed under the terms of the
!  GNU General Public License: see COPYING in the top directory
!  or http://www.gnu.org/copyleft/gpl.txt.
! See Docs/Contributors.txt for a list of contributors.
!
      subroutine dipole( cell, ntm, M1, M2, M3, nsm, rho, X0, dipol )
C ********************************************************************
C Finds the electric dipole
C Written by J.M.Soler. July 1997.
C Modified for distributed rho matrix using a 2D grid of processors.
C Routine now is based on intrinsic structure of grid distribution
C in order to calculate position of grid points in space for local
C matrix. J.D.Gale March 1999.
C *********** INPUT **************************************************
C real*8  cell(3,3)     : Unit cell vectors
C integer ntm(3)        : Global number of divisions of each lattice vector
C integer M1,M2,M3      : Local number of divisions of each lattice vector
C integer nsm           : Number of sub-points for each mesh point
C real    rho(N1,N2,N3) : Minus neutral charge density at mesh points
C                         Notice single precision in this version
C real*8  X0(3)         : Origin in cartesian coordinates
C                         (center of molecule)
C *********** OUTPUT *************************************************
C real*8 dipol(3)   : Electric dipole
C *********** UNITS **************************************************
C cell  in atomic units (Bohr)
C rho   in atomic units (electrons/Bohr**3)
C X0    in atomic units (Bohr)
C dipol in atomic units (electrons*Bohr)
C ********************************************************************
C
C  Modules
C
      use precision,  only : dp, grid_p
      use parallel,   only : Node, Nodes, ProcessorY
      use sys,        only : die
      use mesh,       only : meshLim
#ifdef MPI
      use mpi_siesta
#endif

      implicit none

      integer, intent(in)          ::       M1, M2, M3, ntm(3), nsm
      real(grid_p), intent(in)     ::       rho(M1,M2,M3)
      real(dp), intent(in)         ::       cell(3,3), X0(3)
      real(dp), intent(out)        ::       dipol(3)

      real(dp)  :: volcel
      external          reclat, volcel

C Internal variables and arrays
      integer           I, I1, I2, I3, IX, I10, I20, I30,
     &                  MG1, MG2, MG3, Xoffset, Yoffset, Zoffset
      real(dp)          D(3), dvol, DX, Rcell(3,3), X0L(3)
#ifdef MPI
      integer           MPIerror
#endif

C Assign local variables
      MG1 = ntm(1)
      MG2 = ntm(2)
      MG3 = ntm(3)

C Find volume element
      dvol = volcel( cell ) / (MG1*MG2*MG3)

C Find reciprocal cell vectors (without the factor 2*pi)
      call reclat( cell, Rcell, 0 )

C Find origin in lattice coordinates
      do I = 1,3
        X0L(I)= X0(1)*Rcell(1,I) + X0(2)*Rcell(2,I) + X0(3)*Rcell(3,I)
      enddo

C Initialize dipole
      dipol(1:3) = 0.0_dp

C Calculate starting point for grid
      Xoffset = (meshLim(1,1)-1)*nsm
      Yoffset = (meshLim(1,2)-1)*nsm
      Zoffset = (meshLim(1,3)-1)*nsm

C Find dipole by direct integration allowing for block distributed
C structure of rho
      I30 = Zoffset
      do I3 = 1,M3
        I30 = I30 + 1
        D(3) = dble(I30-1) / dble(MG3) - X0L(3)
        IF (D(3) .LT. - 0.5_dp) D(3) = D(3) + 1.0_dp
        IF (D(3) .GT. + 0.5_dp) D(3) = D(3) - 1.0_dp
        I20 = Yoffset
        do I2 = 1,M2
          I20 = I20 + 1
          D(2) = dble(I20-1) / dble(MG2) - X0L(2)
          IF (D(2) .LT. - 0.5_dp) D(2) = D(2) + 1.0_dp
          IF (D(2) .GT. + 0.5_dp) D(2) = D(2) - 1.0_dp
          I10 = Xoffset
          do I1 = 1,M1
            I10 = I10 + 1
            D(1) = dble(I10-1) / dble(MG1) - X0L(1)
            IF (D(1) .LT. - 0.5_dp) D(1) = D(1) + 1.0_dp
            IF (D(1) .GT. + 0.5_dp) D(1) = D(1) - 1.0_dp
            do IX = 1,3
              DX = cell(IX,1)*D(1) + cell(IX,2)*D(2) + cell(IX,3)*D(3)
              dipol(IX) = dipol(IX) - DX * rho(I1,I2,I3) * dvol
            enddo
          enddo
        enddo
      enddo
#ifdef MPI
      call MPI_AllReduce( dipol, d, 3, MPI_double_precision, MPI_sum,
     &                    MPI_Comm_World, MPIerror )
      dipol(1:3) = d(1:3)
#endif
      end

