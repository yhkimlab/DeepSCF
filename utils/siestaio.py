####                                       ####
#   SEISTA IO Module by RONG                  #
####                                       ####

# Ver. 1.0. Date : 2020/6/17

# Description: SIESTA I/O interface

#--------- Medules list ---------#

import fortranfile as fortran
import struct
import numpy as np
import glob

#---------- Parameters ----------#
bohr2ang = 0.529177249



def readGrid(fname):

    '''
    *************************** VARS **********************************
    character*(*) fname     : File name for input or output
    integer nsm             : Number of sub-mesh points per mesh point
                           (not used in this version)
    integer maxp            : First dimension of array rho
    integer nspin           : Second dimension of array rho
    real*8  cell(3,3)       : Lattice vectors
    integer mesh(3)         : Number of mesh divisions of each
                           lattice vector
    real    rho(nspin, mesh[2], mesh[1], mesh[0]) : Electron density
    '''

    f = fortran.FortranFile(fname)
    cell = f.readReals('d')
    
    cell = np.reshape(cell,(3,3))
    temp = f.readInts('i')

    mesh = temp[:3]
    nspin = temp[3]

    maxp = mesh[0] * mesh[1] * mesh[2]
    rho = np.zeros((nspin, mesh[0], mesh[1], mesh[2]), dtype = float)

    for isp in range(nspin):
        for iz in range(mesh[2]):
            for iy in range(mesh[1]):
                rho[isp,:,iy,iz] = f.readReals('f')

    return cell, mesh, rho


def writeGrid(fname, cell, mesh, rho):

    f = fortran.FortranFile(fname, mode = 'wb')

    cell2 = cell.flatten().tolist()
    f.writeReals(cell2, 'd')

    temp = mesh.tolist()
    nspin = np.shape(rho)[0]
    temp.append(nspin)

    f.writeInts(temp,'i')

    for isp in range(nspin):
        for iz in range(mesh[2]):
            for iy in range(mesh[1]):
                f.writeReals(rho[isp,:,iy,iz],'f')
    f.close()


def readDM(fname):


    '''
    Read unformatted SIESTA sparse matrix file

    Parameters :

        - nb : Number of basis
        - ns : Number of spins
        - ndmax : Total number of nonzero elements

        - numd(nb) : Number of nonzero elements of each row of hamiltonian matrix
        - listhptr(nao) : Pointer to the start of each row of hamiltonian matrix
        - listh(numh) : Nonzero hamiltonian-matrix element column indexes for each matrix row
        - dm(ndmax, ns)
    '''


    f = fortran.FortranFile(fname)

    # basis, spin
    tmp = f.readInts('i')
    nb = tmp[0] 
    ns = tmp[1]

    numd     = np.zeros(nb, dtype = int)
    listdptr = np.zeros(nb, dtype = int)

    numd = f.readInts('i')

    ndmax = 0

    for m in range(nb):
        ndmax = ndmax + numd[m]
        if (m == 0):
          listdptr[m] = 0
        else:
          listdptr[m] = listdptr[m-1] + numd[m-1]

    listd = np.zeros(ndmax, dtype = int)
    dm    = np.zeros((ndmax,ns), dtype = float)


    for m in range(nb):
        n = numd[m]
        listd[listdptr[m]:listdptr[m]+n] = f.readInts('i')

    for isp in range(ns):
        for m in range(nb): 
            n = numd[m]
            dm[listdptr[m]:listdptr[m]+n,isp] = f.readReals('d')
    f.close()


    return nb, ns, numd, listdptr, listd, dm



def writeDM(fname, nb, ns, numd, listdptr, listd, dm):

    f = fortran.FortranFile(fname, mode = 'wb')
    f.writeInts([nb, ns], 'i')
    f.writeInts(numd, 'i')

    for m in range(nb):
        n = numd[m]
        f.writeInts(listd[listdptr[m]:listdptr[m]+n], 'i')

    for isp in range(ns):
        for m in range(nb):
            n = numd[m]
            f.writeReals(dm[listdptr[m]:listdptr[m]+n,isp], 'd')
    f.close()



def readWFSX(fname):

    '''
    Read unformatted SIESTA WFSX file


    Input :

        - fname : WFSX file

    Parameters :

        - nkp : Number of k-points
        - nsp : Number of spins
        - nao : Number of basis orbitals

        - ia(nao) :  Atomic index of each orbital
        - label(nao) : Atomic labels of each orbital
        - iao(nao) : Orbital index of each orbital in each atoms
        - nquant(nao) : Principal quatum number of each orbital
        - sym(nao) : Symmetry of each orbital

        - wk(nkp) : Weight of each k-point
        - pk(3, nkp) : k-point vector
        - eig(nao, nsp, nkp)  : Eigenvalue of 
        - wf(1 or 2, nao, nao, nsp, nkp) : Eigenvector of each 


    '''

    f = fortran.FortranFile(fname)

    # read number of kpoints, spins, atomic orbitals
    nkp, gamma = f.readInts('i')
    nsp = f.readInts('i')[0]
    nao = f.readInts('i')[0]

    # read index of each orbital
    ia = np.zeros((nao), dtype=int)
    label = []
    iao = np.zeros((nao), dtype=int)
    nquant = np.zeros((nao), dtype=int)
    sym = []

    dat = f.readRecord()
    dat_size = struct.calcsize('<i20sii20s') # five values
    ind_st = 0
    ind_fn = dat_size

    for io in range(nao):
        val_list = struct.unpack('<i20sii20s', dat[ind_st:ind_fn])


        ia[io] = val_list[0]
        label.append(val_list[1].decode('ascii').strip())
        iao[io] = val_list[2]
        nquant[io] = val_list[3]
        sym.append(val_list[4].decode('ascii').strip())

        ind_st = ind_st + dat_size
        ind_fn = ind_fn + dat_size

    label = np.array(label)
    sym = np.array(sym)

    # read k compontents, eigenvalue, eigenvector per each k and e
    wk = np.zeros((nkp), dtype=np.float64)
    pk = np.zeros((nkp,3), dtype=np.float64)
    eig = np.zeros((nao, nsp, nkp), dtype=np.float64)

    if (gamma == -1):
        wf = np.zeros((1, nao, nao, nsp, nkp), dtype=float)
    else:
        wf = np.zeros((2, nao, nao, nsp, nkp), dtype=float)


    dat_size = struct.calcsize('<idddd') # problem

    for ik in range(nkp):
        for isp in range(nsp):
            
            dat = f.readRecord()

            val_list = struct.unpack('<idddd', dat[0:dat_size])

            dummy = val_list[0] - 1
            pk[ik,:] = val_list[1:4]
            wk[ik] = val_list[4]


            ispin = f.readInts('i')[0]
            nwf = f.readInts('i')[0]

            if (dummy != ik):
                raise ValueError('ik =! dummy')
            if (ispin > nsp):
                raise ValueError('ispin > nsp')
            if (nwf > nao):
                raise ValueError('nwf > nao')

            for iw in range(nwf):
                iao_ = f.readInts('i')[0] - 1
                eig[iw, isp, ik]  = f.readReals('d')[0]
                buff = f.readReals('f')
                if (gamma == -1):
                    wf[0, :, iw, isp, ik] = buff
                else:
                    buff = buff.reshape((2,-1), order = 'F')
                    wf[:, :, iw, isp, ik] = buff

    f.close()

    return gamma, pk, wk, wf, eig, ia, label, iao, nquant, sym


def readHSX(fname):

    '''
    Read unformatted SIESTA HSX file

    Parameters :

        - nao : Number of basis orbitals per unit cell
        - no_s : Number of basis orbitals per supercell
        - nspin : Spin polarization
        - maxnhtot : non zero
        - gamma : 
        - indxuo(no_s) : Index of equivalant orbital in unit cell

        - numh : Number of nonzero elements of each row of hamiltonian matrix

        - listhptr(nao) : Pointer to the start of each row of hamiltonian matrix
        - listh(numh) : Nonzero hamiltonian-matrix element column indexes for each matrix row

        - hamilt(numx, nspin) : Hamiltonian in sparse form
        - Sover(numx) : Overlap in sparse form
        - xij(3, numx) : Vectors between orbital centers (sparse)

        - qtot : Total number of electrons
        - temp_in_file : Electronic temperature for Fermi smearing

        - nspecies : Total number of different atomic species
        - label(nspecies) : Atomic label for given atomic species
        - zval(nspecies) : Valence charge for given atomic species
        - no(nspecies) : Total number of Basis orbitals for given atomic specie

        - nquant(nspecies, no) : Principal quatum number for a given atomic basis 
        - lquant(nspecies, no) : Total angular momentum quantum number of a given basis orbital
        - zeta(nspecies, no) : Zeta number of a given basis orbital
 
    '''

    f = fortran.FortranFile(fname)
    no_u, no_s, nspin, maxnhtot = f.readInts('i')
    gamma = f.readInts('i')[0]
    
    if gamma ==0:
        indxuo = f.readInts('i')
    else:
        indxuo = np.zeros((no_u), dtype= int)
        for i in range(no_u):
            indxuo[i] = i+1

    numh = f.readInts('i')

    listhptr = np.zeros((no_u,), dtype = int)

    for io in range(1, no_u):
        listhptr[io] = listhptr[io - 1] + numh[io - 1]

    numx = np.max(listhptr)
    ibuff = np.zeros((numx,), dtype = int)
    hbuff = np.zeros((numx,), dtype = float)
    buff3 = np.zeros((numx*3,), dtype = float)

    listh = np.zeros((maxnhtot,), dtype = int)
    hamilt = np.zeros((maxnhtot, nspin), dtype = float)
    Sover = np.zeros((maxnhtot,), dtype = float)
    xij = np.zeros((maxnhtot,3), dtype = float)
    

    for io in range(no_u):
        ptr = listhptr[io]
        n = numh[io]
        ibuff[0:n] = f.readInts('i')
        listh[ptr:ptr + n] = ibuff[0:n] # fortran index

    for isp in range(nspin):
        for io in range(no_u):
            ptr = listhptr[io]
            n = numh[io]
            hbuff[0:n] = f.readReals('f')

            hamilt[ptr:ptr + n, isp] = hbuff[0:n]


    for io in range(no_u):
        ptr = listhptr[io]
        n = numh[io]
        hbuff[0:n] = f.readReals('f')
        Sover[ptr:ptr + n] = hbuff[0:n]

    qtot, temp_in_file = f.readReals('d')

    for io in range(no_u):
        ptr = listhptr[io]
        n = numh[io]
        buff3[0: 3 * n] = f.readReals('f')
        
        for i in range(n):
            xij[ptr+i,0] = buff3[3*i]
            xij[ptr+i,1] = buff3[3*i+1]
            xij[ptr+i,2] = buff3[3*i+2]

    # Read auxiliary info

    nspecies = f.readInts('i')[0]

    label = []
    zval = np.zeros((nspecies,), dtype = np.float64)
    no = np.zeros((nspecies,), dtype = int)

    dat = f.readRecord()
    dat_size = struct.calcsize('<20sdi') # problem
    ind_st = 0
    ind_fn = dat_size

    for ispec in range(nspecies):
        
        val_list = struct.unpack('<20sdi', dat[ind_st:ind_fn])

        label.append(val_list[0].strip())
        zval[ispec] = (val_list[1])
        no[ispec] = val_list[2]

        ind_st = ind_st + dat_size
        ind_fn = ind_fn + dat_size

    nquant = []
    lquant = []
    zeta = []

    for ispec in range(nspecies):
        nquant.append([])
        lquant.append([])
        zeta.append([])
        
        for io in range(no[ispec]):
            abuff, bbuff, cbuff = f.readInts('i')
            
            nquant[-1].append(abuff)
            lquant[-1].append(bbuff)            
            zeta[-1].append(cbuff)
            
    na_u = f.readInts('i')[0] # number of species
    isa = np.zeros((na_u,), dtype = int)
    iaorb = np.zeros((no_u,), dtype = int)
    iphorb = np.zeros((no_u,), dtype = int)

    isa = f.readInts('i')
    
    obuff = np.zeros((2*no_u), dtype = int)
    obuff = f.readInts('i')
    for i in range(no_u):
        iaorb[i] = obuff[i*2]
        iphorb[i] = obuff[i*2+1]

    f.close()

    za = np.zeros((no_u), dtype = int)
    zc = np.zeros((no_u), dtype = int)
    zn = np.zeros((no_u), dtype = int)
    zl = np.zeros((no_u), dtype = int)
    zx = np.zeros((no_u), dtype = int)
    zz = np.zeros((no_u), dtype = int)

    nao = 0
    for ia in range(na_u):
        it = isa[ia]-1 # species
        io = 0
        while(io < no[it]):
            lorb = lquant[it][io]

            for ko in range(lorb*2+1):
                za[nao] = ia + 1 # atomic index
                zc[nao] = it + 1 # atomic species
                zn[nao] = nquant[it][io] # principle quantum number
                zl[nao] = lorb   # total angular momentum quantum number
                zx[nao] = ko + 1 #
                zz[nao] = zeta[it][io]
                nao += 1
            io = io + 2*lorb + 1
           
    return numh, listhptr, listh, indxuo, hamilt, Sover, xij, za, zc, zn, zl, zx, zz


def readDIM(fname):
    '''
    read unformatted SIESTA DIM file

    '''

    f = fortran.FortranFile(fname)

    MAXA = f.readInts('i')[0]
    MAXO = f.readInts('i')[0]
    MAXUO = f.readInts('i')[0]
    NSPIN = f.readInts('i')[0]
    MAXNH = f.readInts('i')[0]
    MAXNA = f.readInts('i')[0]

    f.close()
   
    return MAXA, MAXO, MAXUO, NSPIN, MAXNH, MAXNA


def readPLD(fname, MAXA, MAXO):
    

    '''
    read unformatted SIESTA PLD file

    Input :

       - MAXA
       - MAXO

    Output :

       - IPHORB : Orbital index (within atom) of each orbital
       - INDXUO : Equivalent orbital in unit cell
       - DATM : Occupations of basis orbitals in free atom
       - ISA : Species index of each atom in the supercell
       - LASTO : Last orbital of each atom in array iphorb
       - CELL : Supercell vectors CELL(IXYZ,IVECT) (Bohr)
       - NSC : Num. of unit cells in each supercell direction
       - XA : Atomic positions in cartesian coordinates (Bohr)
    '''


    f = fortran.FortranFile(fname)

    RMAXO = f.readReals('d')[0]

    dat_size = struct.calcsize('<iid')


    IPHORB = np.zeros((MAXO), dtype = int)
    INDXUO = np.zeros((MAXO), dtype = int)
    DATM = np.zeros((MAXO), dtype = np.float64)
    ISA = np.zeros((MAXA), dtype = int)
    LASTO = np.zeros((MAXA+1), dtype = int)
    CELL = np.zeros((3,3), dtype = np.float64) 
    NSC = np.zeros((3), dtype = int)
    XA = np.zeros((3,MAXA), dtype = np.float64)

    for io in range(MAXO):
        dat = f.readRecord()
        val_list = struct.unpack('<iid',dat[0:dat_size])

        IPHORB[io] = val_list[0]
        INDXUO[io] = val_list[1]
        DATM[io] = val_list[2]

    for ia in range(MAXA):
        ISA[ia] = f.readInts('i')[0]

    for ia in range(MAXA+1):
        LASTO[ia] = f.readInts('i')[0]

    for ia in range(3):
        CELL[:,ia] = f.readReals('d')

    NSC = f.readInts('i')

    for ia in range(MAXA):
        XA[:,ia] = f.readReals('d')

    f.close()

    return RMAXO, IPHORB, INDXUO, DATM, ISA, LASTO, ISA, CELL, NSC, XA


def readIon(fname):
    
    pao_basis = {}
    
    with open(fname) as f:

        while(1):
            if f.readline().strip() == '</preamble>':
                break

        symbol = f.readline().split()[0]
        label = f.readline().split()[0]
                    
        f.readline() # atomic number
        f.readline() # valence charge
        f.readline() # mass
        f.readline() # self energy
        atom_info = f.readline()
        atom_info = atom_info.split()
                    
        lmax = int(atom_info[0])
        number_of_nl = int(atom_info[1])
                
        atom_info = {
                    'symbol' : symbol,    
                    'label'  : label,
                    'lmax'   : lmax
                    }
                    
        f.readline() # KB
        f.readline() # strat PAOs
                    
        for iqn in range(number_of_nl):
            line = f.readline()
            word = line.split()
                
            l = int(word[0])
            n = int(word[1])
            z = int(word[2])
            pol = int(word[3])
            pop = float(word[4])
                       
            line = f.readline()
            word = line.split()
                        
            npts = int(word[0])
            delta = float(word[1])
            cutoff = float(word[2]) * bohr2ang # convert unit!
            r = np.zeros((npts), dtype = np.float64)
            phi = np.zeros((npts), dtype = np.float64)
                        
            for ir in range(npts):
                line = f.readline()
                word = line.split()
                            
                r[ir] = np.float64(word[0]) * bohr2ang # convert unit!
                phi[ir] = np.float64(word[1])
                
            if n in pao_basis.keys():
                if l in pao_basis[n].keys():
                    if z in pao_basis[n][l].keys():
                        pao_basis[n][l][z].update({'phi':phi})
                        pao_basis[n][l][z].update({'r':r})
                        pao_basis[n][l][z].update({'cutoff':cutoff})
                    else:
                        pao_basis[n][l].update({z:{}})
                        pao_basis[n][l][z].update({'phi':phi})
                        pao_basis[n][l][z].update({'r':r})
                        pao_basis[n][l][z].update({'cutoff':cutoff})
                else:
                    pao_basis[n].update({l:{}})
                    pao_basis[n][l].update({z:{}})
                    pao_basis[n][l][z].update({'phi':phi})
                    pao_basis[n][l][z].update({'r':r})
                    pao_basis[n][l][z].update({'cutoff':cutoff})
            else:
                pao_basis.update({n:{}})
                pao_basis[n].update({l:{}})
                pao_basis[n][l].update({z:{}})
                pao_basis[n][l][z].update({'phi':phi})
                pao_basis[n][l][z].update({'r':r})
                pao_basis[n][l][z].update({'cutoff':cutoff})

        f.close()
        
    return pao_basis


def readStruct():
    
    struct_file = glob.glob('STRUCT.fdf')[0]

    CELL = np.zeros((3,3), dtype = float)


    with open(struct_file) as f:
        for i, l in enumerate(f):
            line = l
            word = line.split()
            
            if len(word) >= 2:
                if word[0] == 'NumberOfAtoms':
                    number_of_atoms = int(word[1])
                if word[0] == 'NumberOfSpecies':
                    number_of_species = int(word[1])
                if word[0] == 'LatticeConstant':
                    lattice_constant = float(word[1])
                if line.strip() == '%block ChemicalSpeciesLabel':
                    species = np.zeros((number_of_species), dtype =int)
                    for ispec in range(number_of_species):
                        species_line = f.readline()
                        species_words = species_line.split()
                        species[ispec] = int(species_words[1])
                if line.strip() == '%block LatticeVectors':
                    for ix in range(3):
                        cell_line = f.readline()
                        cell_vals = cell_line.split()
                        CELL[ix][0] = cell_vals[0]
                        CELL[ix][1] = cell_vals[1]
                        CELL[ix][2] = cell_vals[2]        
                    CELL = CELL * lattice_constant
                if word[0] == 'AtomicCoordinatesFormat':
                    if word[1] == 'ScaledCartesian':
                        parameter = lattice_constant
                    elif word[1] == 'Ang':
                        parameter = 1
                if line.strip() == '%block AtomicCoordinatesAndAtomicSpecies':
                    SPEC = np.zeros((number_of_atoms), dtype = int)
                    ATOMS = np.zeros((number_of_atoms,3), dtype = float)
                    for ia in range(number_of_atoms):
                        atom_line = f.readline()
                        atom_coord = atom_line.split()
                        ATOMS[ia][0] = atom_coord[0]
                        ATOMS[ia][1] = atom_coord[1]
                        ATOMS[ia][2] = atom_coord[2]
                        SPEC[ia] = species[int(atom_coord[3])-1]
                        
                    ATOMS = ATOMS * parameter

    return CELL, ATOMS, SPEC
