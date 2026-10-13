# Copyright (C) 1996-2016 The SIESTA group
# Distributed under the GNU General Public License; see ../COPYING.
# Serial Linux build using GNU Fortran >= 10 and system BLAS/LAPACK.
# Copy to arch.make after running ../Src/obj_setup.sh.

.SUFFIXES:
.SUFFIXES: .f .F .o .c .a .f90 .F90

SIESTA_ARCH = linux-gfortran-serial
CC = gcc
FC = gfortran
FC_SERIAL = gfortran
FPP = $(FC) -E -P -x c
AR = ar
RANLIB = ranlib

# Legacy SIESTA routines pass arrays of different ranks to some routines.
FFLAGS = -O2 -fallow-argument-mismatch
FFLAGS_DEBUG = -g -O1 -fallow-argument-mismatch
LDFLAGS =
INCFLAGS =
FPPFLAGS = -DFC_HAVE_FLUSH -DFC_HAVE_ABORT

SYS = nag
SP_KIND = 4
DP_KIND = 8
KINDS = $(SP_KIND) $(DP_KIND)
COMP_LIBS =
LIBS = -llapack -lblas
MPI_INTERFACE =
MPI_INCLUDE = .

# Keep the lower optimization level recommended for atom.F.
atom.o: atom.F
	$(FC) -c $(FFLAGS_DEBUG) $(INCFLAGS) $(FPPFLAGS) $<

.c.o:
	$(CC) -c $(CFLAGS) $(INCFLAGS) $(CPPFLAGS) $<
.F.o:
	$(FC) -c $(FFLAGS) $(INCFLAGS) $(FPPFLAGS) $(FPPFLAGS_fixed_F) $<
.F90.o:
	$(FC) -c $(FFLAGS) $(INCFLAGS) $(FPPFLAGS) $(FPPFLAGS_free_F90) $<
.f.o:
	$(FC) -c $(FFLAGS) $(INCFLAGS) $(FCFLAGS_fixed_f) $<
.f90.o:
	$(FC) -c $(FFLAGS) $(INCFLAGS) $(FCFLAGS_free_f90) $<
