! ---
! Copyright (C) 1996-2016	The SIESTA group
!  This file is distributed under the terms of the
!  GNU General Public License: see COPYING in the top directory
!  or http://www.gnu.org/copyleft/gpl.txt .
! See Docs/Contributors.txt for a list of contributors.
! ---
!
      module m_ldau

      use precision, only: dp

      implicit none

      private

      public :: hubbard_term

      ! Last change in population of LDA+U occupations
      real(dp), save, public :: LDAU_dPop = huge(1._dp)
      
      ! These are private variables that is used below
      integer, save :: LDAU_pop_iter = 0

      ! Maximum number of LDA+U projectors
      integer, save :: maxldau = 0
      real(dp), dimension(:,:,:,:), pointer, save :: occu      ! Array used to 
                                                               ! store the 
                                                               ! occupations of
                                                               ! the LDA+U proj.
      real(dp), dimension(:,:,:,:), pointer, save :: occu_old  ! Same as occu
                                                               ! but in the 
                                                               ! previous step

      CONTAINS

      subroutine hubbard_term( scell, nua, na, isa, xa, indxua,
     .                         maxnh, maxnd, lasto, iphorb, no_u, no_l,
     .                         numd, listdptr, listd, numh, 
     .                         listhptr, listh, nspin, Dscf, 
     .                         Eldau, DEldau, Hldau, 
     .                         fa, stress, H, iscf,
     .                         matrix_elements_only )      
C *********************************************************************
C Calculates Hubbard-like U term contribution to total 
C energy, atomic forces, stress and hamiltonian matrix elements.
C Energies in Ry. Lengths in Bohr.
C Merged into the trunk by J. Junquera, March 2016.
C Written by D. Sanchez-Portal, October 2008, 
C after subroutine nlefsm by J.Soler and P.Ordejon (June 1997).
C Based on a previous version by S. Riikonen and D. Sanchez-Portal (2005)
C **************************** INPUT **********************************
C real*8  scell(3,3)       : Supercell vectors SCELL(IXYZ,IVECT)
C integer nua              : Number of atoms in unit cell
C integer na               : Number of atoms in supercell
C integer isa(na)          : Species index of each atom
C real*8  xa(3,na)         : Atomic positions in cartesian coordinates
C integer indxua(na)       : Index of equivalent atom in unit cell
C integer maxnh            : First dimension of H and listh
C integer maxnd            : Maximum number of elements of the
C                            density matrix
C integer lasto(0:na)      : Position of last orbital of each atom
C integer iphorb(no)       : Orbital index of each orbital in its atom,
C                            where no=lasto(na)
C integer no_u             : Number of orbitals in unit cell
C integer no_l             : Number of orbitals (local)
C integer numd(nuo)        : Number of nonzero elements of each row of the
C                            density matrix
C integer listdptr(nuo)    : Pointer to the start of each row (-1) of the
C                            density matrix
C integer listd(maxnd)     : Nonzero hamiltonian-matrix element column 
C                            indexes for each matrix row
C integer numh(nuo)        : Number of nonzero elements of each row of the
C                            hamiltonian matrix
C integer listhptr(nuo)    : Pointer to the start of each row (-1) of the
C                            hamiltonian matrix
C integer listh(maxnh)     : Nonzero hamiltonian-matrix element column 
C                            indexes for each matrix row
C integer nspin            : Number of spin components
C integer iscf             : Counter of the cycles of SCF iterations
C real*8  Dscf(maxnd,nspin): Density matrix
C logical matrix_elements_only: Determine whether only the matrix elements
C                            of the Hamiltonian are computed, or also the
C                            forces and stresses
C ******************* INPUT and OUTPUT *********************************
C real*8 fa(3,na)          : NL forces (added to input fa)
C real*8 stress(3,3)       : NL stress (added to input stress)
C real*8 H(maxnh,nspin)    : NL Hamiltonian (added to input H)
C **************************** OUTPUT *********************************
C real*8 Eldau             : U-Hubbard energy 1 
C real*8 DEldau            : U-hubbard energy 2 
C real*8 Hldau(maxnh,nspin): Hamiltonian elements from LDA+U
C*********************************************************************
C
C  Modules
C
#ifdef MPI
      use m_mpi_utils, only: globalize_sum
#endif

      use parallel,      only : Node, Nodes
      use parallelsubs,  only : GetNodeOrbs, LocalToGlobalOrb
      use parallelsubs,  only : GlobalToLocalOrb
      use atmfuncs,      only : rcut, orb_gindex, ldau_gindex
      use atmfuncs,      only : nofis
      use neighbour    , only : iana=>jan, r2ki=>r2ij, xki=>xij
      use neighbour    , only : mneighb
      use alloc,         only : re_alloc, de_alloc
      use m_new_matel,   only : new_matel
      use radial,        only : rad_func
      use atm_types,     only : nspecies
      use atm_types,     only : species_info    ! Derived type with all the info
                                                !   about the radial functions
                                                !   (PAOs, KB projectors,
                                                !   LDA+U proj,
                                                !   VNA potentials, etc)
                                                !   for a given atomic specie
      use atm_types,   only : species           ! Actual array where the
                                                !   previous information is
                                                !   stored
      use ldau_specs,  only : dDmax_threshold   ! Parameter that defines the
                                                !   criterium required to start
                                                !   or update the calculation of
                                                !   the populations of
                                                !   the LDA+U projections
      use ldau_specs,  only : dtol_ldaupop      ! Parameter that defines the
                                                !   convergence criterium 
                                                !   for the LDA+U local 
                                                !   population
      use ldau_specs,  only : ldau_init         ! Flag that determines whether 
                                                !   the local populations are 
                                                !   calculated on the 
                                                !   first iteration
      use ldau_specs,  only : ldau_shift        ! Flag that determines whether 
                                                !   the U parameter
                                                !   is interpreted
                                                !   as a local potential shift
      use m_compute_max_diff, only: dDmax_current

      integer, intent(in) ::
     .   maxnh, na, maxnd, nspin, nua, iscf, no_u, no_l
      
      integer, intent(in)  ::
     .  indxua(na), iphorb(*), isa(na),  
     .  lasto(0:na), listd(maxnd), listh(maxnh),
     .  numd(no_l), numh(no_l), listdptr(no_l), listhptr(no_l)
      real(dp), intent(inout) :: Hldau(maxnh,nspin)

      real(dp), intent(in) :: scell(3,3), Dscf(maxnd,nspin),
     .                        xa(3,na)
      real(dp), intent(inout) :: fa(3,nua), stress(3,3)
      real(dp), intent(inout) :: H(maxnh,nspin)
      real(dp), intent(out)   :: Eldau, DEldau
      logical, intent(in)     :: matrix_elements_only

      real(dp) :: volcel
      external :: timer, volcel

C Internal variables ................................................
C maxno  = maximum number of basis orbitals overlapping a KB projector
      integer, save :: maxno = 500

      integer
     .  ia, ishldau, ina, ind, ino,
     .  io, iio, ioa, is, ispin, ix, ikb, 
     .  j, jno, jo, jx, ka, ko, ks, kua, ldauidx, 
     .  nldauproj, nna, nno, no, nuotot, ja,
     .  lko, lkb, nprin_ko, nprin_kb, kg, ig 

      integer, dimension(:), pointer :: iano, iono

      real(dp)
     .  Cijk, fik, rki, rmax, rmaxldau, rmaxo, 
     .  Sik, Sjk, volume,  oc(nspin), Ueff, dn, 
     .  Dij, rci

      real(dp), dimension(:,:), pointer :: Vi, Di
      real(dp), dimension(:,:), pointer :: Ski, xno

      real(dp), dimension(:,:,:), pointer :: grSki

#ifdef MPI
      ! Reduction operations
      real(dp), dimension(:,:), pointer :: buffer1 => null()
#endif

      logical :: first                            ! For a given set of 
                                                  !   atomic positions
                                                  !   determine whether this 
                                                  !   is the first step in the
                                                  !   SCF iterations
      logical :: within, pop_conv
      logical, dimension(:), pointer ::  listed, listedall
      logical, save :: firstime = .true.          !  First time that this 
                                                  !   subroutine is called?
      type(species_info),  pointer :: spp
      type(rad_func),      pointer :: pp
C ......................

#ifdef BSC_CELLXC
      ! The only reason is that the XC functional 
      ! is not calculated exactly equivalent.
      ! As such, one may delete this below line to make them accessible 
      ! in conjunction with each other...
      call die('LDA+U and BSC_CELLXC is not compatible')
#endif
      
!     Start time counter
      call timer( 'hubbard_term', 1 )

      ! Nullify pointers
      nullify( grSki, Ski, xno, iono, iano )
      nullify( listedall, listed, Vi, Di )

      ! Make sure the energies are zero
      Eldau = 0.0_dp
      DEldau = 0.0_dp

!     Determine whether this is the first SCF step for a 
!     given atomic configuration
      first = (iscf == 1)
      ! Reset population iteration
      if ( first ) LDAU_pop_iter = 0
         
!     Initialization and allocation of matrices
      if( firstime ) then 
!       Find maximum number of LDA+U projectors on a given atom 
        maxldau = 0
        do ka = 1, na
          is  = isa(ka)
          spp => species(is)
          maxldau = max(maxldau,spp%nprojsldau)
        enddo

!       Allocate local array to store the occupations of the 
!       LDA+U projectors
        nullify( occu, occu_old )
        allocate( occu(maxldau,maxldau,nua,nspin) )
        call memory( 'A', 'D', size(occu), 'hubbard_term' )
        allocate( occu_old(maxldau,maxldau,nua,nspin) )
        call memory( 'A', 'D', size(occu_old), 'hubbard_term' )
        occu_old = 0.0_dp

        firstime  = .false.
      endif

!     End initialization

!     Here we determine whether the occupations are computed on the first
!     SCF step or not.
      if( first .and. (.not. ldau_init) ) then
        if ( Node == 0 ) then
           write(6,'(2a)') 'hubbard_term: ',
     &          'not computing occupations in the first SCF step'
        end if
        call timer( 'hubbard_term', 2 )
        return
      endif

      
      occupations: if( first .or.
     &                 (dDmax_current .lt. dDmax_threshold) .or.
     &                 ldau_shift ) then 


!       Find unit cell volume
        volume = volcel( scell ) * nua / na

!       Find maximum range of the atomic orbitals (rmaxo) 
!       and of the LDA+U projectors (rmaxldau)
        rmaxo    = 0.0_dp
        rmaxldau = 0.0_dp
        do is = 1, nspecies
           
           ! Species orbital range
           do io = 1, nofis(is)
              rmaxo = max(rmaxo, rcut(is,io))
           enddo
           
           ! Species LDAU range
           spp => species(is)
           do io = 1, spp%n_pjldaunl
              pp => spp%pjldau(io)
              rmaxldau = max(rmaxldau, pp%cutoff)
           end do
        enddo
        
        ! Total range of the projector is Oo
        rmax = rmaxo + rmaxldau

!       Initialize arrays Di and Vi only once
        no = lasto(na)
        nuotot = lasto(nua)

!       Allocate local memory
        call re_alloc( Di, 1, no, 1, nspin, 
     &                 name='Di', routine='hubbard_term' )
        Di = 0.0_dp

        call re_alloc( Vi, 1, no, 1, nspin, 
     &                 name='Vi', routine='hubbard_term' )
        Vi = 0.0_dp

        call re_alloc( listed, 1, no, 
     &                 name='listed', routine='hubbard_term')
        listed(1:no) = .false.

        call re_alloc( listedall, 1, no, 
     &                 name='listedall', routine='hubbard_term' )
        listedall(1:no) = .false.

!       Make list of all orbitals needed for this node
        do io = 1, no_l
          call LocalToGlobalOrb(io,Node,Nodes,iio)
          listedall(iio) = .true.
          do j = 1, numh(io)
            jo = listh(listhptr(io)+j)
            listedall(jo) = .true.
          enddo
        enddo

!       Allocate local arrays that depend on saved parameters
        call re_alloc( iano, 1, maxno, 
     &                 name='iano', routine='hubbard_term' )
        call re_alloc( iono, 1, maxno, 
     &                 name='iono', routine='hubbard_term' )
        call re_alloc( xno, 1, 3, 1, maxno, 
     &                 name='xno', routine='hubbard_term' )
        call re_alloc( Ski, 1, maxldau, 1, maxno, 
     &                 name='Ski', routine='hubbard_term' )
        call re_alloc( grSki, 1, 3, 1, maxldau, 1, maxno, 
     &                 name='grSki', routine='hubbard_term' )
      
!       Counter for the SCF loops to converge the population of the 
!       LDA+U projectors
        LDAU_pop_iter = LDAU_pop_iter + 1
        if( Node == 0 ) write(6,'(a,i4)')
     &   'hubbard_term: recalculating local occupations ', LDAU_pop_iter

!       Initialize occupations
        occu = 0.0_dp

!       Initialize neighb subroutine
        call mneighb( scell, rmax, na, xa, 0, 0, nna )

!       Loop on atoms with LDA+U projectors      
        do ka = 1, na
          kua = indxua(ka)
          ks = isa(ka)
          spp => species(ks)
          nldauproj = spp%nprojsldau
          if( nldauproj == 0 ) cycle

!         Find neighbour atoms 
          call mneighb( scell, rmax, na, xa, ka, 0, nna )

!         Find neighbour orbitals
          nno = 0
          do ina = 1, nna
            ia = iana(ina)
            is = isa(ia)
            rki = sqrt(r2ki(ina))
            do io = lasto(ia-1)+1, lasto(ia)

!             Only calculate if needed locally
              if (listedall(io)) then
                ioa = iphorb(io)
                ig  = orb_gindex(is,ioa)
                rci = rcut(is,ioa)

!               Find if orbital is within range
                within = .false.
                do ko = 1, nldauproj
                  ldauidx = spp%pjldau_index(ko)
                  pp => spp%pjldau(ldauidx)
                  if ( rci + pp%cutoff > rki )
     &                 within = .true.
                end do

!               Find overlap between neighbour orbitals and LDA+U projectors
                if (within) then
!                 Check maxno - if too small then increase array sizes
                  if (nno.eq.maxno) then
                    maxno = maxno + 10
                    call re_alloc( iano, 1, maxno, name='iano',
     &                       copy=.true., routine='hubbard_term' )
                    call re_alloc( iono, 1, maxno, name='iono',
     &                       copy=.true., routine='hubbard_term' )
                    call re_alloc( xno, 1, 3, 1, maxno, name='xno', 
     &                       copy=.true., routine='hubbard_term' )
                    call re_alloc( Ski,1, maxldau, 1, maxno, name='Ski',
     &                       copy=.true., routine='hubbard_term' )
                    call re_alloc( grSki, 1, 3, 1, maxldau, 1, maxno,
     &                       name='grSki', routine='hubbard_term', 
     &                       copy=.true. )
                  endif
                  nno = nno + 1
!                 The nno-eme neighbour orbital of the LDA+U projector is io,
!                 where io runs between 1 and the total number of orbitals
!                 in the supercell
                  iono(nno) = io

!                 The nno-eme neighbour orbital of the LDA+U projector belongs
!                 to atom ia,
!                 where ia runs between 1 and the total number of atoms
!                 in the supercell
                  iano(nno) = ia

!                 The relative position between the center of the LDA+U proj.
!                 and the center of the nno-eme neighbour orbital 
!                 is xno 
                  do ix = 1,3
                    xno(ix,nno) = xki(ix,ina)
                  enddo

!                 Here we compute the overlap between a LDA+U projector
!                 and a neighbour atomic orbital
                  do ko = 1, nldauproj
                    kg  = ldau_gindex(ks,ko)
                    call new_matel( 'S', kg, ig, xki(1:3,ina),
     &                  Ski(ko,nno), grSki(1:3,ko,nno) )
                  enddo

                endif   ! If on orbitals within range

              endif     ! If on orbitals within the local node

            enddo       ! End loop on neighbour orbitals

          enddo         ! End loop on neighbour atoms

!         Loop on neighbour orbitals
          do ino = 1,nno
            io = iono(ino)
            ia = iano(ino)
            
            ! Only atoms in the unit cell
            if ( ia > nua ) cycle

            ! Only local orbitals
            call GlobalToLocalOrb(io,Node,Nodes,iio)
            if ( iio < 1 ) cycle
            
!           Scatter filter and density matrix row of orbital io
            do j = 1, numd(iio)
              ind = listdptr(iio) + j
              jo = listd(ind)
              listed(jo) = .true.
              do ispin = 1, nspin
                Di(jo,ispin) = Di(jo,ispin) + Dscf(ind,ispin)
              enddo
            enddo

!           Find matrix elements with other neighbour orbitals
            do jno = 1,nno
              jo = iono(jno)
              ja = iano(jno)
              if ( listed(jo) ) then

!               Loop on LDA+U projectors
                do ko = 1, nldauproj
                  Sik = Ski(ko,ino)
                  do ikb = 1, nldauproj
                    Sjk = Ski(ikb,jno)
                    do ispin = 1, nspin
                      Dij = Di(jo,ispin) 
                      occu(ko,ikb,kua,ispin) = 
     &                     occu(ko,ikb,kua,ispin) + 
     &                     Dij * Sik * Sjk/(3.0_dp-dble(nspin))
                    enddo 
                  enddo
                enddo
              endif
            enddo

!           Restore Di and listed
            do j = 1, numh(iio)
              jo = listh(listhptr(iio)+j)
              listed(jo) = .false.
              do ispin=1,nspin
                 Di(jo,ispin) = 0.0_dp
              enddo
            enddo

          enddo            ! End loop on neighbour orbitals (ino)

        enddo              ! End of the loop on atoms with LDA+U projectors
     
#ifdef MPI
!       Global reduction of occupation
        call re_alloc(buffer1, 1, maxldau, 1, maxldau,
     &       name='buffer1', routine = 'hubbard_term')
        do ka=1,nua
          is=isa(ka)
          spp => species(is)
          nldauproj = spp%nprojsldau
          if( nldauproj == 0 ) cycle
          do ispin=1,nspin
             call globalize_sum(occu(1:nldauproj,1:nldauproj,ka,ispin),
     &            buffer1(1:nldauproj,1:nldauproj))
             occu(1:nldauproj,1:nldauproj,ka,ispin) =
     &            buffer1(1:nldauproj,1:nldauproj)
          end do
        enddo 
        call de_alloc(buffer1, name='buffer1', routine = 'hubbard_term')
#endif

        LDAU_dPop = 0._dp
        do ka=1,nua
          is=isa(ka)
          spp => species(is)
          nldauproj = spp%nprojsldau
          if ( nldauproj == 0 ) cycle
          
          oc = 0.0_dp
          
          if( ldau_shift .and. Node == 0 ) then
             write(6,*) 'hubbard_term: projector occupations'
             write(6,*) 'hubbard_term: atom, species: ',ka, is
          endif
          do ko = 1, nldauproj
            do ikb = 1, nldauproj
              if( ldau_shift .and. Node == 0 ) then
                 write(6,'(2i4,2f12.5)')  ko, ikb,
     &                (occu(ko,ikb,ka,ispin),ispin=1,nspin)
              endif

              do ispin = 1, nspin
                 dn = occu(ko,ikb,ka,ispin)-occu_old(ko,ikb,ka,ispin)
                 LDAU_dPop = max(LDAU_dPop,dabs(dn))
              enddo 
            enddo 
            if ( ldau_shift ) then
              do ispin = 1 , nspin
                oc(ispin) = oc(ispin) + occu(ko,ko,ka,ispin)
              end do
            endif
           enddo 
           if ( ldau_shift .and. Node == 0 ) then
              write(6,'(a,/,a,3f12.6)') 
     &             'hubbard_term: Total projector shell',
     &             'Occupations: ', (oc(ispin),ispin=1,nspin)
     &             ,sum(oc)
           end if
        enddo 

        
#ifdef MPI
        ! LDAU_dPop need not be globalized.
        ! The occupations are already globalized
#endif
        if ( Node == 0 ) then
           write(6,'(a,f12.6)')
     &          'hubbard_term: maximum change in local occup.',LDAU_dPop
        endif
        
        pop_conv = ( LDAU_dPop < dtol_ldaupop )
        if( .not. pop_conv ) occu_old = occu

        recalc_hamilt: if( .not. pop_conv .or. first .or. 
     &                     .not. matrix_elements_only ) then
          if ( Node == 0 ) then 
            if ( matrix_elements_only ) then
              write(6,'(a)')'hubbard_term: recalculating Hamiltonian'
            else
              write(6,'(a)')
     &          'hubbard_term: recalculating Hamiltonian and forces'      
            endif
          endif

          Hldau = 0.0_dp

!         Initialize neighb subroutine
          call mneighb( scell, rmax, na, xa, 0, 0, nna )

          do ka = 1, na
            kua = indxua(ka)
            ks = isa(ka)
            spp => species(ks)
            nldauproj = spp%nprojsldau
            if( nldauproj == 0 ) cycle

!           Find neighbour atoms
            call mneighb( scell, rmax, na, xa, ka, 0, nna )

!           Find neighbour orbitals
            nno = 0
            do ina = 1, nna
              ia = iana(ina)
              is = isa(ia)
              rki = sqrt(r2ki(ina))
              do io = lasto(ia-1)+1, lasto(ia)

!               Only calculate if needed locally
                if (listedall(io)) then
                  ioa = iphorb(io)
                  ig  = orb_gindex(is,ioa)
                  rci = rcut(is,ioa)

!                 Find if orbital is within range
                  within = .false.
                  do ko = 1, nldauproj
                    ldauidx = spp%pjldau_index(ko)
                    pp => spp%pjldau(ldauidx)
                    if ( rci + pp%cutoff > rki )
     &                   within = .true.
                  end do

!                 Find overlap between neighbour orbitals and LDA+U projectors
                  if (within) then
!                   Check maxno - if too small then increase array sizes
                    if (nno.eq.maxno) then
                      maxno = maxno + 10
                      call re_alloc( iano, 1, maxno, name='iano',
     &                         copy=.true., routine='hubbard_term' )
                      call re_alloc( iono, 1, maxno, name='iono',
     &                         copy=.true., routine='hubbard_term' )
                      call re_alloc( xno, 1, 3, 1, maxno, name='xno', 
     &                         copy=.true., routine='hubbard_term' )
                      call re_alloc( Ski,1, maxldau, 1,maxno,name='Ski',       
     &                         copy=.true., routine='hubbard_term' )
                      call re_alloc( grSki, 1, 3, 1, maxldau, 1, maxno,
     &                         name='grSki', routine='hubbard_term', 
     &                         copy=.true. )
                    endif
                    nno = nno + 1
                    iono(nno) = io
                    iano(nno) = ia
                    do ix = 1,3
                      xno(ix,nno) = xki(ix,ina)
                    enddo
                    do ko = 1, nldauproj
                      kg = ldau_gindex(ks,ko)
                      call new_matel( 'S', kg, ig, xki(1:3,ina),
     &                    Ski(ko,nno), grSki(1:3,ko,nno) )
                    enddo

                  endif   ! If on orbitals within range

                endif     ! If on orbitals within the local node

              enddo       ! end loop on neighbour orbitals

            enddo         ! end loop on neighbour atoms


!           Loop on neighbour orbitals
            do ino = 1,nno
              io = iono(ino)
              ia = iano(ino)

              ! Only atoms in the unit cell
              if ( ia > nua) cycle

              ! Only local orbitals
              call GlobalToLocalOrb(io,Node,Nodes,iio)
              if ( iio < 1 ) cycle

!             Scatter filter/density matrix row of orbital io
              do j = 1,numd(iio)
                ind = listdptr(iio)+j
                jo = listd(ind)
                listed(jo) = .true.
                do ispin = 1,nspin
                   Di(jo,ispin) = Di(jo,ispin) + Dscf(ind,ispin)
                enddo
              enddo

!             Find matrix elements with other neighbour orbitals
              do jno = 1,nno
                jo = iono(jno)
                ja = iano(jno)

                if ( listed(jo) ) then
!                 Loop on LDA+U projectors
                  do ko = 1, nldauproj
                    lko      = spp%pjldau_l(ko)
                    nprin_ko = spp%pjldau_n(ko)
                    ldauidx  = spp%pjldau_index(ko)
                    Ueff     = spp%pjldaunl_U(ldauidx) -
     &                         spp%pjldaunl_J(ldauidx)
                    if( ldau_shift ) Ueff = 2.0_dp * Ueff
                    Sik = Ski(ko,ino)
                    Sjk = Ski(ko,jno)
                    do ispin=1, nspin
                      Vi(jo,ispin) = Vi(jo,ispin) +
     &                               0.5_dp * Sik * Sjk * Ueff
                      Dij  = Di(jo,ispin)
                      Cijk = Ueff * Dij * Sjk 
                      if(.not. matrix_elements_only) then 
                        do ix=1,3
                          fik = Cijk * grSki(ix,ko,ino)
                          fa(ix,ia)  = fa(ix,ia)  - fik
                          fa(ix,kua) = fa(ix,kua) + fik
                          do jx = 1, 3
                            stress(jx,ix) = stress(jx,ix) +
     &                                      xno(jx,ino)*fik/volume       
                          enddo 
                        enddo 
                      endif

                      if( .not. ldau_shift ) then 
                        do ikb = 1, nldauproj
                          lkb      = spp%pjldau_l(ikb)
                          nprin_kb = spp%pjldau_n(ikb)
                          if( lko      .eq. lkb        .and.
     &                        nprin_ko .eq. nprin_kb ) then 
!                            For the time being we will use the formulation
!                            of Dudarev and collaborators
                             Cijk = occu(ko,ikb,kua,ispin) *
     &                              Ueff * Ski(ikb,jno)
                             Vi(jo,ispin) = Vi(jo,ispin) -
     &                                      Sik * Cijk
                             if(.not. matrix_elements_only) then 
                              do ix=1,3
                                fik = -2.0_dp * Dij * Cijk * 
     &                                grSki(ix,ko,ino) 
                                fa(ix,ia)  = fa(ix,ia)  - fik
                                fa(ix,kua) = fa(ix,kua) + fik
                                do jx = 1, 3
                                  stress(jx,ix)= stress(jx,ix) +
     &                                    xno(jx,ino) * fik / volume
                                enddo
                              enddo 
                            endif
                          endif 
                        enddo 
                      endif

                    enddo        ! Enddo ispin
                  enddo     ! Enddo ko, on LDA+U projectors
                endif       ! Endif if the orbital jo is listed
              enddo         ! Enddo on neighbour orbitals jno

!             Add to Hldau and restore Di, Vi and listed
              do j = 1,numh(iio)
                ind = listhptr(iio)+j
                jo = listh(ind)
                do ispin = 1,nspin
                  Hldau(ind,ispin) = Hldau(ind,ispin) + 
     &                               Vi(jo,ispin)
                  Vi(jo,ispin) = 0.0_dp
                  Di(jo,ispin) = 0.0_dp
                enddo 
                listed(jo) = .false.
              enddo

            enddo        ! enddo ino = 1,nno
          enddo          ! enddo loop on orbitals with LDA+U projectors(ka=1,na)

        endif recalc_hamilt

      endif occupations

      add_hamilt: if ( LDAU_pop_iter > 0 ) then 
        
        ! Add the Hamiltonian elements and calculate energy
        do iio = 1 , no_l
           do j = 1, numh(iio)
              ind = listhptr(iio) + j
              do ispin = 1, nspin
                 Eldau = Eldau +
     &                0.5_dp * Hldau(ind,ispin) * Dscf(ind,ispin)       
                 H(ind,ispin) = H(ind,ispin) + Hldau(ind,ispin)
              end do
           end do
        end do
        
        do ia = 1,nua
          is = isa(ia)
          spp => species(is)
          nldauproj = spp%nprojsldau
          if( nldauproj .gt. 0 ) then
            do ispin = 1, nspin
              do ko = 1, nldauproj
                ldauidx = spp%pjldau_index(ko)
                Ueff    = spp%pjldaunl_U(ldauidx) -
     &                    spp%pjldaunl_J(ldauidx)
                if( ldau_shift ) Ueff = 2.0_dp * Ueff
                DEldau = DEldau +
     &               0.25_dp * (3.0_dp-dble(nspin)) * Ueff *
     &               occu(ko,ko,ia,ispin)
              enddo
            enddo   
          endif
        enddo

      endif add_hamilt

!      Deallocate local memory

      call de_alloc( grSki, name='grSki' )
      call de_alloc( Ski, name='Ski' )
      call de_alloc( xno, name='xno' )
      call de_alloc( iono, name='iono' )
      call de_alloc( iano, name='iano' )

      call de_alloc( listedall, name='listedall' )
      call de_alloc( listed, name='listed' )
      call de_alloc( Vi, name='Vi' )
      call de_alloc( Di, name='Di' )

!     Stop time counter
      call timer( 'hubbard_term', 2 )

      end subroutine hubbard_term
!
      end module m_ldau
