! 
! Copyright (C) 1996-2016	The SIESTA group
!  This file is distributed under the terms of the
!  GNU General Public License: see COPYING in the top directory
!  or http://www.gnu.org/copyleft/gpl.txt.
! See Docs/Contributors.txt for a list of contributors.
!
      subroutine ioeig(eo, ef, no, nspin, nk, maxo, nspinor, maxk,
     .                 kpoints, kweights)

c *******************************************************************
c Writes eigenvalues of Hamiltonian in k-points of sampling
c Emilio Artacho, Feb. 1999

      use precision, only : dp
      use siesta_cml
      use units, only : eV
      use files, only : slabel, label_length

      implicit          none

      integer,  intent(in) :: no
      integer,  intent(in) :: nspin
      integer,  intent(in) :: nk
      integer,  intent(in) :: maxo
      integer,  intent(in) :: nspinor
      integer,  intent(in) :: maxk
      real(dp), intent(in) :: ef
      real(dp), intent(in) :: eo(maxo, nspinor, maxk)
      real(dp), intent(in) :: kpoints(3,nk)
      real(dp), intent(in) :: kweights(nk)
      
      external          io_assign, io_close

c Internal 
      integer           ik, iu, io, is

      character(len=label_length+4) :: fname
c -------------------------------------------------------------------

      fname = trim(slabel) // '.EIG'
      
      call io_assign( iu )
      open( iu, file=fname, form='formatted', status='unknown' )      

      write(iu,"(e17.9)") ef/eV
      if ( nspin > nspinor ) then
        write(iu,"(tr1,i10,i2,tr1,i10)")   no*2, 1, nk
      else
        write(iu,"(tr1,i10,i2,tr1,i10)")   no, min(nspin,2), nk
      end if
      do ik = 1,nk
        write(iu,"(i10,10(tr1,e17.9),/,(tr10,10(tr1,e17.9)))")
     .          ik, ((eo(io,is,ik)/eV,io=1,no),is=1,nspinor)
      enddo

      call io_close( iu )

      if (cml_p) then
         call cmlStartPropertyList(xf=mainXML, title="Eigenvalues")
         call cmlAddProperty(xf=mainXML, value=ef/eV, 
     .        title='Fermi Energy', dictref='siesta:E_Fermi', 
     .        fmt='r5', units='siestaUnits:ev')
         call cmlAddProperty(xf=mainXML, value=nk, 
     .        title='Number of k-points', dictRef='siesta:nkpoints',
     .        units='cmlUnits:countable')
         if ( nspin > nspinor ) then
           call cmlStartPropertyList(mainXML, dictRef='siesta:kpt_band')
           do ik = 1, nk
             call cmlAddKPoint(xf=mainXML, coords=kpoints(:, ik), 
     .             weight=kweights(ik))
             call cmlAddProperty(xf=mainXML,
     .           value=reshape(eo(1:no,1:nspinor,ik)/eV,
     .           (/no*nspinor/)),
     .            dictRef='siesta:eigenenergies',
     .            units='siestaUnits:ev')
           end do
           call cmlEndPropertyList(mainXML)
         else
          do is = 1 , nspinor
            call cmlStartPropertyList(mainXML,
     .           dictRef='siesta:kpt_band')
            if ( is == 1 .and. nspinor > 1 ) then
               call cmlAddProperty(xf=mainXML, value="up", 
     .              dictRef="siesta:spin")
            else if ( nspinor > 1 ) then
               call cmlAddProperty(xf=mainXML, value="down", 
     .              dictRef="siesta:spin")
            end if
            do ik = 1, nk
               call cmlAddKPoint(xf=mainXML, coords=kpoints(:, ik), 
     .              weight=kweights(ik))
               call cmlAddProperty(xf=mainXML, value=eo(1:no,is,ik)/eV, 
     .              dictRef='siesta:eigenenergies',
     .              units='siestaUnits:ev')
            enddo
            call cmlEndPropertyList(mainXML)
          end do
         end if
         call cmlEndPropertyList(mainXML)
      endif
      
      end subroutine ioeig
