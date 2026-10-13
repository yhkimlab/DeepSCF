! ---
! Copyright (C) 1996-2016	The SIESTA group
!  This file is distributed under the terms of the
!  GNU General Public License: see COPYING in the top directory
!  or http://www.gnu.org/copyleft/gpl.txt .
! See Docs/Contributors.txt for a list of contributors.
! ---
!
!     Some of the functionality of the former write_subs module 
!     has been split into  three sub-modules to work around a compiler bug.
!     The submodules appear first in this file, and module
!     write_subs is at the end.
!
      Module write_subs_positions

      use m_steps, only: final

      private
      public :: siesta_write_positions
      CONTAINS
      
      subroutine siesta_write_positions(moved)
      use siesta_geom
      use atomlist, only: elem, iza
      use siesta_cml
      use units, only: Ang
      use zmatrix, only: lUseZmatrix, write_canonical_ucell_and_Zmatrix
      use m_iostruct, only: write_struct
      use siesta_options, only: idyn

      implicit none

!     Whether the structure has already changed in response to forces and stresses
      logical, intent(in) :: moved

!
!      Write out structural information in "crystallography" format,
!      and in "canonical" (see Zmatrix module) zmatrix format.


        call write_struct( ucell, na_u, isa, iza, xa, moved)
        if (lUseZmatrix .and. (idyn .eq. 0)) then
           if (moved) then
              call write_canonical_ucell_and_Zmatrix
     $                         (filename="NEXT_ITER.UCELL.ZMATRIX")
           else
              call write_canonical_ucell_and_Zmatrix
     $                         (filename="OUT.UCELL.ZMATRIX")
           endif
        endif

        ! Note that this information should be written only at the
        ! time of processing the current geometry (in state_init)

        if ((.not. moved) .and. cml_p) then
          call cmlAddMolecule(xf=mainXML, natoms=na_u, elements=elem,
     .           atomRefs=cisa, coords=xa/Ang)
          call cmlAddLattice(xf=mainXML, cell=ucell/Ang, 
     .           units='siestaUnits:Ang', dictref='siesta:ucell')
        endif

      end subroutine siesta_write_positions

      END MODULE write_subs_positions
      Module write_subs_pressure

      use m_steps, only: final

      private
      public :: siesta_write_stress_pressure

      CONTAINS
      
      subroutine siesta_write_stress_pressure()

      use parallel, only: IOnode
      use precision
      USE siesta_options
      use siesta_geom
      use atomlist, only: iza
      use m_iostruct,   only: write_struct
      use siesta_cml
      use units
      use m_spin
      use m_energies, only: FreeE
      use m_stress, only: stress, kin_stress, mstress, cstress, tstress

      implicit none

      integer :: jx, ix

      real(dp):: Pmol     ! Molecular pressure (discounting Virial term)
      real(dp):: Psol     ! Pressure of "solid"
      real(dp):: Press    ! Pressure
      real(dp):: ps(3,3)  ! Auxiliary array

! Stress tensor and pressure:
      
      if (.not.final) then
!
!           Write Voigt components of total stress tensor 
!
            ps = stress + kin_stress
            write(6,'(/,a,6f12.2)')
     .           'Stress-tensor-Voigt (kbar):',
     .           (ps(jx,jx)/kbar,jx=1,3),
     $            ps(1,2)/kbar,
     $            ps(2,3)/kbar,
     $            ps(1,3)/kbar
            Press = - ((ps(1,1) + ps(2,2) + ps(3,3))/3.0_dp)
            write(6,"(a,f14.4)") "(Free)E + p*V (eV/cell)",
     $           (FreeE + Press*volume_of_some_cell)/eV

            if (RemoveIntraMolecularPressure) then
               ps = mstress + kin_stress
               write(6,'(/,a,6f12.2)')
     .           'Inter-Molecular-Stress-Voigt (kbar):',
     .           (ps(jx,jx)/kbar,jx=1,3),
     $            ps(1,2)/kbar,
     $            ps(2,3)/kbar,
     $            ps(1,3)/kbar
               Press = - ((ps(1,1) + ps(2,2) + ps(3,3))/3.0_dp)
               write(6,"(a,f14.4)")
     $              "(Free)E + p_inter_molec * V  (eV/cell)",
     $              (FreeE + Press*volume_of_some_cell)/eV
            endif
!
!      This use of the volume is OK, as it is called from state_analysis,
!      before possibly changing the cell.
!      Write "target enthalpy" (E + pV, where p is the *target* pressure)
            write(6,"(a,f14.4)") "Target enthalpy (eV/cell)",
     $           (FreeE + tp*volume_of_some_cell)/eV

      ! Write stress to CML file always
        if (cml_p) then
           call cmlAddProperty(xf=mainXML, value=stress*Ang**3, 
     .             dictref='siesta:stress', title='Stress',
     .             units='siestaUnits:evpa3')
        endif         !cml_p
        
      ! Output depends on dynamics option
        select case (idyn)
        case(0:5,8,9)
           
          if (idyn==0 .and. (.not.varcel)) then
            continue
          else
            write(6,'(/,a,3(/,a,3f12.6))')
     .           'siesta: Stress tensor (static) (eV/Ang**3):',
     .           ('     ',(stress(jx,ix)*Ang**3/eV,jx=1,3),ix=1,3)
            Psol = - ((stress(1,1) + stress(2,2) + stress(3,3))/3.0_dp)
            write(6,'(/,a,f20.8,a)')
     .           'siesta: Pressure (static):', Psol/kBar, '  kBar'
!            write(6,'(/,a,3(/,a,3f12.6))')
!     .       'siesta: Stress tensor (static-constrained) (eV/Ang**3):',
!     .           ('     ',(cstress(jx,ix)*Ang**3/eV,jx=1,3),ix=1,3)
!            write(6,'(a,f20.8,a)')
!     .       'siesta: Pressure (static-constrained):',
!     .       - ((cstress(1,1)+cstress(2,2)+cstress(3,3))/3.0_dp)/kBar,
!     .       '  kBar'
            if (cml_p) then
              ! stress written above
              call cmlAddProperty(xf=mainXML, value=Psol, 
     .             dictref='siesta:psol', title='Pressure (Static)',
     .             units='siestaUnits:kBar')
            endif                !cml_p
            write(6,'(/,a,3(/,a,3f12.6))')
     .           'siesta: Stress tensor (total) (eV/Ang**3):',
     .           ('     ',(tstress(jx,ix)*Ang**3/eV,jx=1,3),ix=1,3)
            Psol = - ((tstress(1,1)+tstress(2,2) +tstress(3,3))/3.0_dp)
            write(6,'(/,a,f20.8,a)')
     .           'siesta: Pressure (total):', Psol/kBar, '  kBar'
            if (cml_p) then
              call cmlAddProperty(xf=mainXML, value=tstress*Ang**3, 
     .             dictref='siesta:tstress', title='Total Stress',
     .             units='siestaUnits:evpa3')
              call cmlAddProperty(xf=mainXML, value=Psol,
     .             dictref='siesta:tpsol', title='Pressure (Total)',
     .             units='siestaUnits:kBar')
            endif !cml_p

            if (RemoveIntraMolecularPressure) then
             ps = mstress
             write(6,'(/,a,3(/,a,3f12.6))')
     .           'siesta: Stress tensor (nonmol) (eV/Ang**3):',
     .           ('     ',(ps(jx,ix)*Ang**3/eV,jx=1,3),ix=1,3)
             Psol = - ((ps(1,1) + ps(2,2) + ps(3,3))/3.0_dp)
             write(6,'(/,a,f20.8,a)')
     .           'siesta: Pressure (nonmol):', Psol/kBar, '  kBar'
             if (cml_p) then
              call cmlAddProperty(xf=mainXML, value=ps*Ang**3, 
     .          dictref='siesta:mstress', 
     .          title='Stress tensor (normal)',
     .          units='siestaUnits:evpa3')
              call cmlAddProperty(xf=mainXML, value=Psol, 
     .          dictref='siesta:pmol', title='Pressure (Nonmol)',
     .          units='siestaUnits:kBar')
             endif                !cml_p
!
             ps = mstress + kin_stress
             write(6,'(/,a,3(/,a,3f12.6))')
     .           'siesta: Stress tensor (nonmol+kin) (eV/Ang**3):',
     .           ('     ',(ps(jx,ix)*Ang**3/eV,jx=1,3),ix=1,3)
             Psol = - ((ps(1,1)+ps(2,2) +ps(3,3))/3.0_dp)
             write(6,'(/,a,f20.8,a)')
     .           'siesta: Pressure (nonmol+kin):', Psol/kBar, '  kBar'
             if (cml_p) then
              call cmlAddProperty(xf=mainXML, value=ps*Ang**3, 
     .             dictref='siesta:tmstress', 
     .             title='Stress tensor (nonmol+kin)',
     .             units='siestaUnits:evpa3')
              call cmlAddProperty(xf=mainXML, value=Psol,
     .         dictref='siesta:tpmol', title='Pressure (Nonmol+Kin)',
     .         units='siestaUnits:kBar')
             endif                !cml_p
            endif                 ! Remove intramolecular pressure

          endif                  !varcel

        end select !idyn

      else !final

           ! AG: Possible BUG 
           ! The volume here refers to the "old" cell, and
           ! might be out of date if the cell has changed.

! Print stress tensor unconditionally
        write(6,'(/,a,3(/,a,3f12.6))')
     .       'siesta: Stress tensor (static) (eV/Ang**3):',
     .       ('siesta: ',(stress(jx,ix)*Ang**3/eV,jx=1,3),ix=1,3)
        if (cml_p) then
          call cmlAddProperty(xf=mainXML, value=stress*Ang**3/eV, 
     .         dictref='siesta:stress', units='siestaUnits:eV_Ang__3')
        endif !cml_p

! Print constrained stress tensor if different from unconstrained
        if (Any(cstress /= stress )) then
             write(6,'(/,a,3(/,a,3f12.6))')
     .       'siesta: Constrained stress tensor (static) (eV/Ang**3):',
     .       ('siesta: ',(cstress(jx,ix)*Ang**3/eV,jx=1,3),ix=1,3)
          if (cml_p) then
            call cmlAddProperty(xf=mainXML, value=cstress*Ang**3/eV, 
     .           dictref='siesta:cstress', 
     .           units='siestaUnits:eV_Ang__3')
          endif !cml_p
        endif

! Find pressure

        Psol = - (( stress(1,1) + stress(2,2) + stress(3,3) )/3.0_dp)
        Pmol = - (( mstress(1,1) + mstress(2,2) + mstress(3,3) )/3.0_dp)

        write(6,'(/,a,f18.6,a)')
     .       'siesta: Cell volume =',
     $       volume_of_some_cell/Ang**3, ' Ang**3'
        write(6,'(/,a,/,a,2a20,a,3(/,a,2f20.8,a))')
     .       'siesta: Pressure (static):',
     .       'siesta: ','Solid',        'Molecule',      '  Units',
     .       'siesta: ', Psol,           Pmol,           '  Ry/Bohr**3',
     .       'siesta: ', Psol*Ang**3/eV, Pmol*Ang**3/eV, '  eV/Ang**3',
     .       'siesta: ', Psol/kBar,      Pmol/kBar,      '  kBar'
        if (cml_p) then
          call cmlStartPropertyList(mainXML, title='Final Pressure')
          call cmlAddProperty(xf=mainXML,
     $         value=volume_of_some_cell/Ang**3, 
     .         title='cell volume', dictref='siesta:cellvol', 
     .         units='siestaUnits:Ang__3')
          call cmlAddProperty(xf=mainXML, value=Psol/kBar, 
     .         title='Pressure of Solid', dictref='siesta:pressSol', 
     .         units='siestaUnits:kbar')
          call cmlAddProperty(xf=mainXML, value=Pmol/kBar,       
     .         title='Pressure of Molecule', dictref='siesta:pressMol', 
     .         units='siestaUnits:kbar')
          call cmlEndPropertyList(mainXML)
        endif !cml_p
        
      endif !final for stress & pressure

      end subroutine siesta_write_stress_pressure
      
      
      End Module write_subs_pressure

      Module write_subs_energies

      use m_steps, only: final, istp

      private
      public :: siesta_write_energies
      CONTAINS
      
      subroutine siesta_write_energies( iscf, dDmax, dHmax )
      USE siesta_options
      use siesta_cml
      use units
      use m_energies 
      use m_spin
      use m_ts_global_vars, only: TSinit, TSrun
      implicit none

      integer              :: iscf
      real(dp), intent(in) :: dDmax ! Max. change in dens. matrix elem.
      real(dp), intent(in) :: dHmax ! Max. change in H elements

      character(len=64) :: fmt
      character(len=6) :: scf_name

      logical              :: first_scf_step
      integer              :: i

      scf_name = 'scf'
      if ( TSrun ) then
         scf_name = 'ts-scf'
      end if
      
      first_scf_step = (iscf == 1)
      ! Only print out full decomposition at very beginning and end.
      if ((istp==1.and.first_scf_step).or.final) then
        write(6,'(/,a,/,(a,f17.6))')
     .     'siesta: Program''s energy decomposition (eV):',
     .     'siesta: Ebs     =', Ebs/eV, 
     .     'siesta: Eions   =', Eions/eV,
     .     'siesta: Ena     =', Ena/eV,
     .     'siesta: Ekin    =', Ekin/eV,
     .     'siesta: Enl     =', Enl/eV,
     .     'siesta: Eso     =', Eso/eV,
     .     'siesta: Eldau   =', Eldau/eV,
     .     'siesta: Dxc     =', Dxc/eV,
     .     'siesta: DEna    =', DEna/eV,
     .     'siesta: DUscf   =', DUscf/eV,
     .     'siesta: DUext   =', DUext/eV,
     .     'siesta: Enegf   =', DE_NEGF/eV,
     .     'siesta: Exc     =', Exc/eV,
     .     'siesta: eta*DQ  =', Ecorrec/eV,
     .     'siesta: Emadel  =', Emad/eV,
     .     'siesta: Emeta   =', Emeta/eV,
     .     'siesta: Emolmec =', Emm/eV,
     .     'siesta: Ekinion =', Ekinion/eV,
     .     'siesta: Eharris =', (Eharrs+Ekinion)/eV,
     .     'siesta: Etot    =', (Etot+Ekinion)/eV,
     .     'siesta: FreeEng =', (FreeE+Ekinion)/eV
        if (cml_p) then
          call cmlStartPropertyList(mainXML,
     .         title='Energy Decomposition')
          call cmlAddProperty(xf=mainXML, value=Ebs/eV, 
     .         units='siestaUnits:eV', 
     .         dictref='siesta:Ebs', fmt='r6')
          call cmlAddProperty(xf=mainXML, value=Eions/eV, 
     .         units='siestaUnits:eV', 
     .         dictref='siesta:Eions', fmt='r6')
          call cmlAddProperty(xf=mainXML, value=Ena/eV,
     .         units='siestaUnits:eV', 
     .         dictref='siesta:Ena', fmt='r6')
          call cmlAddProperty(xf=mainXML, value=Ekin/eV, 
     .         units='siestaUnits:eV', 
     .         dictref='siesta:Ekin', fmt='r6')
          call cmlAddProperty(xf=mainXML, value=Enl/eV, 
     .         units='siestaUnits:eV', 
     .         dictref='siesta:Enl', fmt='r6')
          call cmlAddProperty(xf=mainXML, value=Eldau/eV,
     .         units='siestaUnits:eV',
     .         dictref='siesta:Eldau', fmt='r6')
          call cmlAddProperty(xf=mainXML, value=DEna/eV, 
     .         units='siestaUnits:eV', 
     .         dictref='siesta:DEna', fmt='r6')
          call cmlAddProperty(xf=mainXML, value=Eso/eV, 
     .         units='siestaUnits:eV', 
     .         dictref='siesta:Eso', fmt='r6')
          call cmlAddProperty(xf=mainXML, value=DUscf/eV, 
     .         units='siestaUnits:eV',
     .         dictref='siesta:DUscf', fmt='r6')
          call cmlAddProperty(xf=mainXML, value=DUext/eV, 
     .         units='siestaUnits:eV', 
     .         dictref='siesta:DUext', fmt='r6')
          call cmlAddProperty(xf=mainXML, value=DE_NEGF/eV, 
     .         units='siestaUnits:eV', 
     .         dictref='siesta:Enegf', fmt='r6')
          call cmlAddProperty(xf=mainXML, value=Exc/eV, 
     .         units='siestaUnits:eV', 
     .         dictref='siesta:Exc', fmt='r6')
          call cmlAddProperty(xf=mainXML,value=Ecorrec/eV,
     .         units='siestaUnits:eV', 
     .         dictref='siesta:Ecorrec', fmt='r6')
          call cmlAddProperty(xf=mainXML, value=Emad/eV, 
     .         units='siestaUnits:eV', 
     .         dictref='siesta:Emad', fmt='r6')
          call cmlAddProperty(xf=mainXML, value=Emeta/eV, 
     .         units='siestaUnits:eV', 
     .         dictref='siesta:Emeta', fmt='r6')
          call cmlAddProperty(xf=mainXML, value=Emm/eV, 
     .         units='siestaUnits:eV', 
     .         dictref='siesta:Emm', fmt='r6')
          call cmlAddProperty(xf=mainXML,value=Ekinion/eV,
     .         units='siestaUnits:eV', 
     .         dictref='siesta:Ekinion', fmt='r6')
          call cmlAddProperty(xf=mainXML, value=(Eharrs+Ekinion)/eV,
     .         units='siestaUnits:eV', 
     .         dictref='siesta:EharrsK', fmt='r6')
          call cmlAddProperty(xf=mainXML, value=(Etot+Ekinion)/eV, 
     .         units='siestaUnits:eV', 
     .         dictref='siesta:EtotK', fmt='r6')
          call cmlAddProperty(xf=mainXML, value=(FreeE+Ekinion)/eV, 
     .         units='siestaUnits:eV', 
     .         dictref='siesta:FreeEK', fmt='r6')
          call cmlEndPropertyList(mainXML)
        endif
      endif
!     On all SCF steps, print out the current energy
      if (.not.final) then
!       Print total energy and density matrix error 
        if (cml_p) then
          call cmlStartPropertyList(mainXML, title='SCF Cycle')
!         Eharrs is always output
          call cmlAddProperty(xf=mainXML, value=Eharrs/eV, 
     .         units="siestaUnits:eV", 
     .         dictRef="siesta:Eharrs", fmt="r7")
          if ( .not. harrisfun ) then
             ! store dDmax and dHmax
             call cmlAddProperty(xf=mainXML, value=dDmax, 
     .            units="siestaUnits:none", 
     .            dictRef="siesta:dDmax", fmt="r7")
             call cmlAddProperty(xf=mainXML, value=dHmax/eV,
     .            units="siestaUnits:eV", 
     .            dictRef="siesta:dHmax", fmt="r7")
          end if
        endif
       
!       Determines which properties are output.
        if (harrisfun) then 
          write(6,"(/a,f14.6,/)") 'siesta: Eharris(eV) = ', Eharrs/eV
!         No need for further cml output
        elseif ((isolve==SOLVE_DIAGON) .or. (isolve==SOLVE_PEXSI)
     .                              .or. (isolve==SOLVE_TRANSI)) then
          if (cml_p) then
             call cmlAddProperty(xf=mainXML, value=Etot/eV, 
     .             units="siestaUnits:eV", 
     .             dictRef="siesta:Etot", fmt="r7")
             call cmlAddProperty(xf=mainXML, value=FreeE/eV, 
     .             units="siestaUnits:eV", 
     .             dictRef="siesta:FreeE", fmt="r7")
             if ( fixspin ) then
                call cmlAddProperty(xf=mainXML, value=Efs(1)/eV, 
     .               units="siestaUnits:eV", 
     .               dictRef="siesta:Ef_UP", fmt="r7")
                call cmlAddProperty(xf=mainXML, value=Efs(2)/eV, 
     .               units="siestaUnits:eV", 
     .               dictRef="siesta:Ef_DN", fmt="r7")
                call cmlAddProperty(xf=mainXML, value=Efs(:)/eV, 
     .               units="siestaUnits:eV", 
     .               dictRef="siesta:Efs")
             else
                call cmlAddProperty(xf=mainXML,value=Ef/eV, 
     .               units="siestaUnits:eV", 
     .               dictRef="siesta:Ef", fmt="r7")
             end if
          endif

          if ( fixspin ) then
             if ( (iscf == 1) .or. muldeb ) then
                write(6,'(/,a12,3a16,4a10)')
     .           'iscf', 'Eharris(eV)', 'E_KS(eV)', 'FreeEng(eV)', 
     .           'dDmax', 'Ef_up', 'Ef_dn(eV)','dHmax(eV)'
             end if
             write(6,'(a8,i4,3f16.6,4f10.6)')
     .  trim(scf_name)//': ',iscf, Eharrs/eV, Etot/eV, FreeE/eV, 
     .           dDmax, Efs(1:2)/eV, dHmax/eV
          else                  ! fixspin
             if ( (iscf == 1) .or. muldeb ) then
                write(6,'(/,a12,3a16,3a10)')
     .           'iscf', 'Eharris(eV)', 'E_KS(eV)', 'FreeEng(eV)', 
     .           'dDmax','Ef(eV)','dHmax(eV)'
             end if
             write(6,'(a8,i4,3f16.6,3f10.6)')
     .  trim(scf_name)//': ',iscf, Eharrs/eV, Etot/eV, FreeE/eV, 
     .           dDmax, Ef      /eV, dHmax/eV
          end if                ! fixspin
          
       elseif ((isolve==SOLVE_ORDERN) .or. (isolve==SOLVE_MINIM)) then
          
          write(6,'(/,a15,i4)') 'siesta: iscf = ',iscf
          write(6,'(a14,f15.4,a13,f15.4,a10,f10.6/)') 
     .         'Eharris(eV) = ',Eharrs/eV,
     .         'E_KS(eV) = ',Etot/eV,'dDmax = ',dDmax
          if (cml_p) then
            call cmlAddProperty(xf=mainXML, value=Etot/eV, 
     .           units="siestaUnits:eV", 
     .           dictRef="siesta:Etot", fmt="r7")
         endif
         
       endif                     !harrisfun/isolve

        if (cml_p) then
          call cmlEndPropertyList(mainXML)
        endif
        
      else !final
!       Print out additional information in finalization.
        write(6,'(/,a)') 'siesta: Final energy (eV):'
        write(6,'(a,a15,f17.10)')
     .    'siesta: ', 'Band Struct. =', Ebs/eV,
     .    'siesta: ',      'Kinetic =', Ekin/eV,
     .    'siesta: ',      'Hartree =', Uscf/eV,
     .    'siesta: ',      'Eldau   =', Eldau/eV,
     .    'siesta: ',      'Eso     =', Eso/eV,
     .    'siesta: ',   'Ext. field =', DUext/eV,
     .    'siesta: ',      'Enegf   =', DE_NEGF/eV,
     .    'siesta: ',  'Exch.-corr. =', Exc/eV,
     .    'siesta: ', 'Ion-electron =', (Enascf+Enl+DUscf-Uscf-Uatm)/eV,
     .    'siesta: ',      'Ion-ion =', (Ena+Uatm-Enaatm-Eions)/eV,
     .    'siesta: ',      'Ekinion =', Ekinion/eV,
     .    'siesta: ',        'Total =', (Etot+Ekinion)/eV
        if ( fixspin ) then
           write(6,'(a,a15,f15.6)')
     .          'siesta: ', 'Fermi_up =', Efs(1)/eV,
     .          'siesta: ', 'Fermi_dn =', Efs(2)/eV
        else
           write(6,'(a,a15,f15.6)')
     .          'siesta: ', 'Fermi =', Ef/eV
        end if
        if (cml_p) then
          call cmlStartPropertyList(xf=mainXML, title='Final Energy')
          call cmlAddProperty(xf=mainXML, value=Ebs/eV,  
     .         units='siestaUnits:eV', 
     .         dictref='siesta:Ebs', fmt='r7')
          call cmlAddProperty(xf=mainXML, value=Ekin/eV,  
     .         units='siestaUnits:eV', 
     .         dictref='siesta:Ekin', fmt='r7')
          call cmlAddProperty(xf=mainXML, value=Uscf/eV,  
     .         units='siestaUnits:eV', 
     .         dictref='siesta:Uscf', fmt='r7')
          call cmlAddProperty(xf=mainXML, value=Eldau/eV,  
     .         units='siestaUnits:eV', 
     .         dictref='siesta:Eldau', fmt='r7')
          call cmlAddProperty(xf=mainXML, value=Eso/eV,
     .         units='siestaUnits:eV', 
     .         dictref='siesta:Eso', fmt='r7')
          call cmlAddProperty(xf=mainXML, value=DUext/eV, 
     .         units='siestaUnits:eV', 
     .         dictref='siesta:DUext', fmt='r7')
          call cmlAddProperty(xf=mainXML, value=DE_NEGF/eV, 
     .         units='siestaUnits:eV', 
     .         dictref='siesta:Enegf', fmt='r7')
          call cmlAddProperty(xf=mainXML, value=Exc/eV,   
     .         units='siestaUnits:eV', 
     .         dictref='siesta:Exc', fmt='r7')
          call cmlAddProperty(xf=mainXML, 
     .         value=(Enascf+Enl+DUscf-Uscf-Uatm)/eV, 
     .         units='siestaUnits:eV', 
     .         dictref='siesta:I-e', fmt='r7')
          call cmlAddProperty(xf=mainXML, 
     .         value=(Ena+Uatm-Enaatm-Eions)/eV,
     .         units='siestaUnits:eV', 
     .         dictref='siesta:I-I', fmt='r7')
          call cmlAddProperty(xf=mainXML, value=Ekinion/eV,
     .         units='siestaUnits:eV', 
     .         dictref='siesta:Ekinion', fmt='r7')
          call cmlAddProperty(xf=mainXML, value=(Etot+Ekinion)/eV,
     .         units='siestaUnits:eV', 
     .         dictref='siesta:Etot', fmt='r7')
          call cmlEndPropertyList(mainXML)
        endif !cml_p
      endif !final

      end subroutine siesta_write_energies

      END MODULE write_subs_energies
!
!     Main module
!
      Module write_subs

      use write_subs_pressure,  only : siesta_write_stress_pressure
      use write_subs_energies,  only : siesta_write_energies
      use write_subs_positions, only : siesta_write_positions

      private
      public :: siesta_write_forces, siesta_write_stress_pressure,
     &          siesta_write_energies, siesta_write_positions
      CONTAINS
      
      subroutine siesta_write_forces(istep)
      use parallel, only: IOnode
      USE siesta_options
      use siesta_geom
      use siesta_cml
      use units,      only : Ang, eV
      use m_forces
      use m_steps, only: final, inicoor
      use alloc,      only : re_alloc, de_alloc
      implicit none
      integer, intent(in) :: istep
      integer :: ia, ix
      real(dp) :: cfmax, fmax, fres
      real(dp) :: ftot(3), cftot(3)

#ifdef DEBUG
      call write_debug( '    PRE siesta_write_forces' )
#endif
      fmax = maxval(abs(fa))
      ftot = sum(fa, dim=2)
      cfmax = maxval(abs(cfa))
      cftot = sum(cfa, dim=2)
      fres = sqrt( sum(fa**2) / (na_u*3.0_dp))

      ! Almost the same forces output whether during simulation
      ! or at the end. Unfortunately not quite, therefore slightly
      ! tortuous logic below. If we are content to change format
      ! of output file slightly, this can be simplified.
      if (.not.final) then
        ! print forces to xml every step. 
        ! output forces to stdout depending on writef
        if (cml_p) then
          call cmlStartPropertyList(mainXML, title='Forces')
          call cmlAddComment(mainXML,"Output: matrix fa(1:3,1:na_u)") 
          call cmlAddProperty(xf=mainXML, value=fa*Ang/eV,
     .         dictref='siesta:forces', units='siestaUnits:evpa')
          call cmlAddProperty(xf=mainXML, value=ftot*Ang/eV,
     .         dictref='siesta:ftot', units='siestaUnits:evpa')
          call cmlAddProperty(xf=mainXML, value=fmax*Ang/eV, 
     .         dictref='siesta:fmax', units='siestaUnits:evpa')
          call cmlAddProperty(xf=mainXML, value=fres*Ang/eV,
     .         dictref='siesta:fres', units='siestaUnits:evpa')
          call cmlAddProperty(xf=mainXML, value=cfmax*Ang/eV, 
     .         dictref='siesta:cfmax', units='siestaUnits:evpa')
          call cmlEndPropertyList(mainXML)
        endif

        write(6,'(/,a)') 'siesta: Atomic forces (eV/Ang):'
        if (writef) then
          write(6,'(i6,3f12.6)')(ia,(fa(ix,ia)*Ang/eV,ix=1,3),ia=1,na_u)
        endif

        ! Always write forces to .FA file
        call iofa( na_u, fa , .false.)
        if ( any(fa/=cfa) ) then
           ! if any one constraint is not the same
           ! as the real forces, the FAC file 
           ! will also be created
           call iofa( na_u, cfa , .true. )
        end if

        write(6,'(40("-"),/,a6,3f12.6)') 'Tot',(ftot(ix)*Ang/eV,ix=1,3)
        write(6,'(40("-"),/,a6, f12.6)') 'Max',fmax*Ang/eV
        write(6,'(a6,f12.6,a)')'Res',fres*Ang/eV,
     .       '    sqrt( Sum f_i^2 / 3N )'
        write(6,'(40("-"),/,a6, f12.6,a)') 'Max',cfmax*Ang/eV, 
     .       '    constrained'

        ! Write Force Constant matrix if FC calculation ...
        select case (idyn)
        case(6)
           ! If the istep is the first step, then it
           ! must be the first 
           call ofc(fa,dx,na_u,.false.,istep==inicoor)
           if ( any(fa/=cfa) ) then
              call ofc(cfa,dx,na_u,.true.,istep==inicoor)
           end if
        case(7)
!          call phonon_write_forces(fa,na_u,ucell,istep)
           if (IOnode) write(*,*) 'phonon support deactivated'
        end select

      else !not final
! In finalization, only print forces if sufficiently large.
        if (fmax .gt. ftol) then
          write(6,'(/,a)') 'siesta: Atomic forces (eV/Ang):'
          write(6,'(a,i6,3f12.6)')
     .         ('siesta: ', ia,(fa(ix,ia)*Ang/eV,ix=1,3),ia=1,na_u)
          write(6,'(a,40("-"),/,a,a6,3f12.6)')
     .         'siesta: ','siesta: ','Tot',(ftot(ix)*Ang/eV,ix=1,3)
          if (cml_p) then
            call cmlStartPropertyList(mainXML, title='Force Summary')
            call cmlAddComment(mainXML,"Output: matrix fa(1:3,1:na_u)") 
            call cmlAddProperty(xf=mainXML, value=fa*Ang/eV,
     .           dictref='siesta:forces', units='siestaUnits:evpa')
            call cmlAddProperty(xf=mainXML, value=ftot*Ang/eV, 
     .           dictref='siesta:ftot', units='siestaUnits:evpa')
            call cmlEndPropertyList(mainXML)
          endif !cml_p
        endif
        if (Any(cfa /= fa)) then
          if (cfmax .gt. ftol) then
            write(6,'(/,a)') 'siesta: Constrained forces (eV/Ang):'
            write(6,'(a,i6,3f12.6)')
     .           ('siesta: ',ia,(cfa(ix,ia)*Ang/eV,ix=1,3),ia=1,na_u)
            write(6,'(a,40("-"),/,a,a4,3f12.6)')
     .           'siesta: ','siesta: ','Tot',(cftot(ix)*Ang/eV,ix=1,3)
            if (cml_p) then
              call cmlStartPropertyList(mainXML,
     .             title='Constrained Force Summary')
              call cmlAddComment(mainXML,
     $             "Output: matrix cfa(1:3,1:na_u)") 
              call cmlAddProperty(xf=mainXML, value=cfa*Ang/eV, 
     .             dictref='siesta:cforces', units='siestaUnits:evpa')
              call cmlAddProperty(xf=mainXML, value=cftot*Ang/eV, 
     .             dictref='siesta:cftot', units='siestaUnits:evpa')
              call cmlEndPropertyList(mainXML)
            endif !cml_p
          endif
        endif
      endif !final for forces
#ifdef DEBUG
      call write_debug( '    POS siesta_write_forces' )
#endif

      end subroutine siesta_write_forces

      End Module write_subs
