! ---
! Copyright (C) 1996-2016	The SIESTA group
!  This file is distributed under the terms of the
!  GNU General Public License: see COPYING in the top directory
!  or http://www.gnu.org/copyleft/gpl.txt .
! See Docs/Contributors.txt for a list of contributors.
! ---

! This module can be used as a place-holder for
! creation citations in a generic way.
! Nick Papior, 2016 February.
!
! USAGE:
! To initialize using a citation table you *must*
! call the routine:
!  if ( IONode ) then
!   call init_citation(<prefix>[, delete=T/F])
!  end if
! This routine will do two things:
!   1. store the filename of the file as:
!        <prefix>//".bib"
!      Hence the citations that are needed to be cited
!      can subsequent to the run be found in the
!      corresponding file.
!   2. It will delete the file without checking the
!      contents.
!      This ensures that the file is always consistent
!      with the latest run.
!      This deletion may be controlled by adding
!      a second, logical, argument (default = .TRUE.)
!
! Subsequently you may call add_citation
! to add specific citations.
! For instance to add the SIESTA-paper as a citation
! you should add this block:
!  if ( IONode ) then
!   call add_citation("10.1088/0953-8984/14/11/302")
!  end if
! This will first check that the citation haven't
! already been added, and if so it will be added
! to the .bib file.
!
! Developers:
! To add a new citation you *MUST* follow these steps:
!  1. Increment citation parameter counter:
!     N_citations = N_citations + 1
!  2. Add a case block with proper information
!     I would highly suggest to copy one existing
!     block and correct the "case ( ... )" statement.
!  3. Correct the "used(ID)" to the top number.
!     I.e. if N_citations were 2 before step 1.
!     Then ID is now 3. (the N_citations value after
!     step 1).
!     Be sure to correct this in the "case" construct.
!
! This code is fully decoupled from the SIESTA
! code base which makes it equally usable in any utility
! which also encourages a collaborative citation
! style in utilities.
!
! This is a very minimal method of providing citation
! cababilities. However, it functions and may
! easily be converted to more complex behaviour if
! one so desires.
! For instance one may imagine that an environment
! variable may be used to find a complete
! BiBTeX file and the code instead finds the information
! in the corresponding file.
module m_cite

  implicit none

  ! DEV-NOTE
  ! Increment this after having added a new
  ! citation!
  ! OTHERWISE YOU WILL EXPERIENCE A SEG-FAULT.
  integer, parameter :: N_citations = 7

  private

  character(len=*), parameter :: STR_NULL = 'NULL'
  integer, parameter :: LEN_COMMENT = 256

  integer, parameter :: LEN_TYPE = 32
  integer, parameter :: LEN_DOI = 64
  integer, parameter :: LEN_CITEKEY = 64
  integer, parameter :: LEN_TITLE = 256
  integer, parameter :: LEN_AUTHOR = 512
  integer, parameter :: LEN_JOURNAL = 128
  integer, parameter :: LEN_VOLUME = 32
  integer, parameter :: LEN_ISSUE = 32

  type citation
     character(len=LEN_COMMENT) :: comment = STR_NULL

     ! regular bib-tex entry table data
     character(len=LEN_TYPE) :: type = 'article'
     character(len=LEN_AUTHOR) :: author = STR_NULL
     character(len=LEN_TITLE) :: title = STR_NULL
     character(len=LEN_JOURNAL) :: journal = STR_NULL
     integer :: year = 0
     character(len=LEN_VOLUME) :: volume = STR_NULL
     character(len=LEN_ISSUE) :: issue = STR_NULL
     character(len=LEN_CITEKEY) :: cite_key = STR_NULL
     character(len=LEN_DOI) :: doi = STR_NULL

  end type citation

  ! Default file-name
  character(len=64) :: cite_file = "CITATIONS.bib"

  ! Simple way to save (a little) memory
  integer, parameter :: INT_KIND = selected_int_kind(r=1)
  integer(INT_KIND), save :: used(N_citations) = 0

  public :: init_citation, add_citation
  
contains

  function get_unit() result(u)
    integer :: u
    logical :: opened

    u = 99
    opened = .true.
    do while ( opened )
       u = u + 1
       inquire( unit=u , opened = opened )
    end do
    
  end function get_unit
    

  subroutine init_citation(pre, delete)
    character(len=*), intent(in) :: pre
    logical, intent(in), optional :: delete
    integer :: iu
    logical :: ldelete

    ldelete = .true.
    if ( present(delete) ) ldelete = delete

    cite_file = trim(pre) // ".bib"
    iu = get_unit()

    if ( .not. ldelete ) return
    
    ! Be sure to have an "empty" citation file
    open( iu , file = cite_file, &
         form = 'formatted', &
         position = 'APPEND')

    ! delete file
    close( iu, status='DELETE' )
         
  end subroutine init_citation

  ! Adds a citation
  ! The `key` is first searched in the doi targets
  ! and then subsequently 
  subroutine add_citation(key)
    character(len=*), intent(in) :: key
    type(citation) :: cit
    integer :: ID

    ID = 0

    select case ( key )

    case ( "10.1088/0953-8984/14/11/302" )
       ! Siesta paper
       cit%comment = "Primary SIESTA paper"
       cit%doi = "10.1088/0953-8984/14/11/302"
       cit%journal = "Journal of Physics: Condensed Matter"
       cit%year = 2002
       cit%volume = "14"
       cit%issue = "11"
       cit%cite_key = "Soler2002"
       ID = 1
       
    case ( "10.1103/PhysRevB.65.165401" )
       ! transiesta paper
       cit%comment = "Primary TranSiesta paper"
       cit%doi = "10.1103/PhysRevB.65.165401"
       cit%journal = "Physical Review B"
       cit%year = 2002
       cit%volume = "65"
       cit%issue = "16"
       cit%cite_key = "Brandbyge2002"
       ID = 2
       
    case ( "10.1103/PhysRevB.59.12301" )
       ! slap dipole correction
       cit%comment = "Slab-dipole correction"
       cit%doi = "10.1103/PhysRevB.59.12301"
       cit%journal = "Physical Review B"
       cit%year = 1999
       cit%volume = "59"
       cit%issue = "16"
       cit%cite_key = "Bengtsson1999"
       ID = 3

    case ( "10.1103/PhysRevB.57.1505" )
       ! implementation specific LDA+U
       cit%comment = "LDA+U implementation"
       cit%doi = "10.1103/PhysRevB.57.1505"
       cit%journal = "Physical Review B"
       cit%year = 1998
       cit%volume = "57"
       cit%issue = "3"
       cit%cite_key = "Dudarev1998"
       ID = 4

    case ( "10.1039/C5CP04613K" )
       ! charge/hartree gate
       cit%comment = "Charge/Hartree gate model"
       cit%doi = "10.1039/C5CP04613K"
       cit%journal = "Phys. Chem. Chem. Phys."
       cit%year = 2016
       cit%volume = "18"
       cit%issue = "2"
       cit%cite_key = "Papior2016A"
       ID = 5

    case ( "10.1016/j.cpc.2016.09.022" )
       ! new transiesta article
       cit%comment = "TranSiesta N-electrode"
       cit%doi = "10.1016/j.cpc.2016.09.022"
       cit%journal = "Computer Physics Communications"
       cit%year = 2017
       cit%volume = "212"
       cit%cite_key = "Papior2017"
       ID = 6

    case ( "10.1088/0953-8984/26/30/305503" )
       ! PEXSI-siesta
       cit%comment = "SIESTA-PEXSI"
       cit%doi = "10.1088/0953-8984/26/30/305503"
       cit%journal = "Journal of Physics: Condensed Matter"
       cit%year = 2014
       cit%volume = "26"
       cit%issue = "30"
       cit%cite_key = "Lin2014"
       ID = 7
       
    end select

    ! probably we should error out...
    ! but...
    if ( ID == 0 ) return
    
    ! Actually create citation
    if ( used(ID) == 0 ) then
       used(ID) = 1
       call write_citation(cit)
    end if

  end subroutine add_citation

  subroutine write_citation(cit)
    type(citation), intent(in) :: cit
    integer :: iu
    
    iu = get_unit()
    
    open( iu , file = cite_file, &
         form = 'formatted', &
         position = 'APPEND')

    ! Add initial comment if this is the first time
    ! being called.
    if ( sum(used) == 1 ) then
       write(iu, '(a)') '# This file contains the papers &
            &that you should cite in case of publishing a paper.'
       write(iu, '(a)') '# Each entry corresponds to using a &
            &feature that has been enabled via FDF-flags'
       write(iu, '(a)') '# and which is based on a development &
            &paper as indicated.'
       write(iu, *)
    end if

    if ( cit%comment /= STR_NULL ) then
       write(iu, '(2a)') '# ', trim(cit%comment)
    end if
    write(iu, '(5a)') '@',trim(cit%type),'{',trim(cit%cite_key),','
    if ( cit%author /= STR_NULL ) then
       write(iu, '(t3,3a)') &
            'author = {{',trim(cit%author),'}},'
    end if
    if ( cit%title /= STR_NULL ) then
       write(iu, '(t3,3a)') &
            'title = {{',trim(cit%title),'}},'
    end if
    if ( cit%journal /= STR_NULL ) then
       write(iu, '(t3,3a)') &
            'journal = {{',trim(cit%journal),'}},'
    end if
    if ( cit%year /= 0 ) then
       write(iu, '(t3,a,i0,a)') &
            'year = {',cit%year,'},'
    end if
    if ( cit%volume /= STR_NULL ) then
       write(iu, '(t3,3a)') &
            'volume = {',trim(cit%volume),'},'
    end if
    if ( cit%issue /= STR_NULL ) then
       write(iu, '(t3,3a)') &
            'issue = {',trim(cit%issue),'},'
    end if
    if ( cit%doi /= STR_NULL ) then
       write(iu, '(t3,3a)') &
            'doi = {',trim(cit%doi),'},'
    end if
    write(iu, '(a)') '}'
    write(iu, *) !

    close( iu )
    
  end subroutine write_citation
  
end module m_cite
