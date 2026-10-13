#!/usr/bin/env python

from __future__ import print_function
# Should make it "backwards" compatible down to 2.6

from tbt_tb import PeriodicTable, SIESTA_UNITS
from tbt_tb import TBT_Geom, TBT_Model, TBT_dH, TBInputFile, TBOutputFile

# This utility function creates a phonon transport input
# to phtrans.

import numpy as np

# It is _exactly_ the same as tbt_tb with some small modifications
# to allow some differentations.

class PHT_Geom(TBT_Geom):
    """
    Geometry object for phonon geometries
    Very similar to the TBT_Geom, however with the change
    that all atoms default to 3 orbitals (x/y/z)

    See ``TBT_Geom`` for specifications.
    """
    def __init__(self,cell,xa,dR=None,n_orb=3,Z=1,update_sc=False,iter=15):
        super(PHT_Geom, self).__init__(cell,xa,dR=dR,n_orb=n_orb,Z=Z,update_sc=update_sc,iter=iter)

    

class PHT_Model(TBT_Model):
    _GEOM = PHT_Geom
    """
    Inherits TBT_Model to create a phonon tight-binding model.

    See ``TBT_Model`` for specifications.
    """

    # TODO, do this in another way, with this object ALL energies
    # are transferred to Ry ** 2
    Ry = SIESTA_UNITS.Ry ** 2

    def correct_Newton(self):
        """
        Sometimes the dynamical matrix does not obey Newtons laws.

        We correct the dynamical matrix by imposing zero force.

        Correcting for Newton forces the matrix to be finalized.
        """
        from scipy.sparse import lil_matrix

        # We need to ensure that the block diagonal exists
        # to create the Newton-corrections
        # This will ONLY work if the matrix has not been 
        # finalized, OR if the block diagonal already exists
        for ja in range(self.na_u):
            jo = ja * 3
            for j in range(3):
                for i in range(3):
                    H, S = self[jo+j,jo+i]
                    if j == i:
                        self[jo+j,jo+i] = H, 1.
                    else:
                        self[jo+j,jo+i] = H, S

        # Create UC dynamical matrix
        d_sc, S_sc = self.tocsr() ; del S_sc
        d_sc = d_sc.tocoo()
        d_uc = lil_matrix((self.no_u,self.no_u),dtype=np.float)

        # Convert SC to UC
        for j, i, d in zip(d_sc.row,d_sc.col,d_sc.data):
            d_uc[j, i % self.no_u] += d
        del d_sc
        d_uc = d_uc.tocsr()

        # we need to correct the dynamical matrix found in GULP
        # This ensures that Newtons laws are obeyed, (i.e. 
        # action == re-action)
        ptbl = PeriodicTable()
        om = ptbl.atomic_mass(self.geom.Z)
        del ptbl

        for ja in range(self.na_u):

            # Create conversion to force-constant, and revert back
            # after correcting
            am = om[ja]
            MM = np.sqrt(am * om)
            jo = ja * 3

            for j in range(3):
                for i in range(3):
                    H, S = self[jo+j,jo+i]
                    self[jo+j,jo+i] = H - d_uc[jo+j,i::3].multiply(MM).sum() / am, S

        del d_uc
        

class PHInputFile(TBInputFile):
    def read_geom(self,cls=PHT_Geom):
        return super(PHInputFile, self).read_geom(cls=cls)

    def read_model(self,geom=None,dtype=np.float,cls=PHT_Model):
        if geom is None: geom = self.read_geom()
        return super(PHInputFile, self).read_model(geom=geom,dtype=dtype,cls=cls)


class GULP(PHInputFile):
    """ Class that extracts information from an GULP output """
    def read_model(self,geom=None,dtype=np.float,cls=PHT_Model):
        """ 
        Returns a new dynamical matrix in the following lines
        Always returns a coo matrix.
        """
        if not hasattr(self,'fh'):
            # The file-handle has not been opened
            with self as fh:
                return fh.read_model(geom=geom,dtype=dtype,cls=cls)

        if geom is None:
            geom = self.read_geom(cls=cls._GEOM)

        from scipy.sparse import lil_matrix, diags

        no = geom.na_u * 3
        dyn = lil_matrix( (no,no) ,dtype = dtype)

        self.step_to('Real Dynamical matrix')

        # skip 1 line
        self.readline()
        
        l = self.readline()
        i = 0 ; j = 0
        while len(l.strip()) > 0:
            ls = [float(f) for f in l.split()]
            # add the values (12 values == 3*4)
            for k in range(4):
                dyn[i,j  ] = ls[k*3  ]
                dyn[i,j+1] = ls[k*3+1]
                dyn[i,j+2] = ls[k*3+2]
                j += 3
                if j >= no: 
                    i += 1
                    j = 0
                    if i >= no: break
            l = self.readline()

        if dtype in [np.complex,np.complex64,np.complex128]:

            # Find the imaginary part
            self.step_to('Imaginary Dynamical matrix')

            # skip 1 line
            self.readline()
            
            l = self.readline()
            i = 0 ; j = 0
            while len(l.strip()) > 0:
                ls = np.array([float(f) for f in l.split()])
                # add the values
                for k in range(4):
                    dyn[i,j  ] += 1j*ls[k*3  ]
                    dyn[i,j+1] += 1j*ls[k*3+1]
                    dyn[i,j+2] += 1j*ls[k*3+2]
                    j += 3
                    if j >= no: 
                        i += 1
                        j = 0
                        if i >= no: break

                l = self.readline()

        # Convert the GULP data to standard units
        dyn = dyn.tocoo()
        dyn.data[:] *= ( 521.469 * 1.23981e-4 ) ** 2

        S = diags([1.],[0],shape=dyn.shape)
        S = S.tocsr()

        return cls.sparse2model(geom,dyn,S)


    def read_geom(self,keyword="Final fractional coordinates",cls=PHT_Geom):
        """
        Returns a ``PHT_Geom`` by reading a the assigned output file.

        Parameters
        ----------
        keyword: str
             keyword to look in the output file after having read the
             lattice vectors.
             If `fractional` is found in the keyword it will expect it
             to be in fractional coordinates. Else it will be read
             in _as is_.
        """
        if not hasattr(self,'fh'):
            # The file-handle has not been opened
            with self as fh:
                return fh.read_geom(keyword=keyword,cls=cls)

        self.step_to('Cartesian lattice vectors')

        # skip 1 line
        self.readline()
        cell = np.zeros((3,3),np.float)
        for i in range(3):
            l = self.readline().split()
            cell[i,0] = float(l[0])
            cell[i,1] = float(l[1])
            cell[i,2] = float(l[2])
            
        # Skip to keyword
        self.step_to(keyword)
                    
        # We skip 5 lines
        for i in range(5): self.readline()
        
        Z = []
        xa = []
        l = self.readline()
        while l[0] != '-':
            ls = l.split()
            Z.append(ls[1])
            xa.append([float(f) for f in ls[3:6]])
            l = self.readline()

        # Convert to array
        xa = np.array(xa,np.float)
        xa.shape = (-1,3)
        if 'fractional' in keyword.lower():
            # Correct for fractional coordinates
            xa[:,0] *= np.sum(cell[:,0])
            xa[:,1] *= np.sum(cell[:,1])
            xa[:,2] *= np.sum(cell[:,2])
            
        if len(Z) == 0 or len(xa) == 0:
            raise ValueError('Could not read in cell information and/or coordinates')

        # Return the geometry
        geom = cls(cell,xa,Z=Z,n_orb=3)
        geom.update(nsc=np.zeros((3,),np.int))
        return geom
