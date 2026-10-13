! 
! Copyright (C) 1996-2016	The SIESTA group
!  This file is distributed under the terms of the
!  GNU General Public License: see COPYING in the top directory
!  or http://www.gnu.org/copyleft/gpl.txt.
! See Docs/Contributors.txt for a list of contributors.
!
      subroutine readsp(q, spiral)
C *********************************************************************
C Subroutine to read the data for the SIESTA program
C
C     It uses the FDF (Flexible Data Fromat) package 
C     of J.M.Soler and A.Garcia
C
C Writen by J. Ferrer. June 2002
C **************************** OUTPUT *********************************
C real*8 q(3)              : Pitch wave vector for spiral configuration
C logical spiral           : True = Spiral arrangement of spins
C **********************************************************************

C
C  Modules
C
      use precision
      use parallel,   only : Node
      use sys,        only : die
      use fdf

      implicit none

      real(dp)
     .  q(3)

      logical
     .  spiral

      external
     .  redcel

C  Internal variables .................................................

      character
     .  lattice*30

      integer 
     .  mscell(3,3)

      logical
     .  lSpinSpiral

      real(dp)
     .  pi, alat, ucell(3,3), scell(3,3), rcell(3,3)

      type(block_fdf)            :: bfdf
      type(parsed_line), pointer :: pline

C Read lattice constant and unit cell vectors .........................

      call redcel( alat, ucell, scell, mscell )
      lSpinSpiral = fdf_block('SpinSpiral',bfdf)
C...

C Spiral arrangement for spins ........................................
      pi = 2.d0 * asin(1.d0)

      if ( lSpinSpiral ) then
        spiral = .true.
        if (.not. fdf_bline(bfdf,pline))
     .    call die('readsp: ERROR in SpinSpiral block')
        lattice = fdf_bnames(pline,1)
        q(1) = fdf_bvalues(pline,1)
        q(2) = fdf_bvalues(pline,2)
        q(3) = fdf_bvalues(pline,3)
        if ( lattice .eq. 'Cubic')  then
          q(1) = pi * q(1) / alat
          q(2) = pi * q(2) / alat
          q(3) = pi * q(3) / alat
        else if ( lattice .eq. 'ReciprocalLatticeVectors' ) then
          call reclat( ucell, rcell, 1 )
          q(1) = rcell(1,1)*q(1) + rcell(1,2)*q(2) + rcell(1,3)*q(3)
          q(2) = rcell(2,1)*q(1) + rcell(2,2)*q(2) + rcell(2,3)*q(3)
          q(3) = rcell(3,1)*q(1) + rcell(3,2)*q(2) + rcell(3,3)*q(3)
        else
          call die('redata: ERROR: ReciprocalCoordinates must be' //
     .             ' ''Cubic'' or ''ReciprocalLatticeVectors''')
        endif
        call fdf_bclose(bfdf)
      else
        spiral = .false.
        q(1) = 0.d0
        q(2) = 0.d0
        q(3) = 0.d0
        lattice = 'unpolarized or collinear spins'
      endif

c     write(6,*) 'redata: Basis for vector q:    ', lattice
c     write(6,'(a,4(f9.2,3x))')
c    .  'redata: q(3)    ', q(1), q(2), q(3)

C...

      return
      end
