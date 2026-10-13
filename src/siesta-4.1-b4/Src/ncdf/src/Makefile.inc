# Define src-directory
src_DIR = $(TOP_DIR)/src
VPATH := $(VPATH):$(src_DIR)


# Define sources in this src directory
src_OBJECTS = netcdf_ncdf.o
src_SOURCES = $(patsubst %.o, %.f90, $(src_OBJECTS))
src_FPP_SOURCES = $(src_SOURCES)
src_MODS = $(patsubst %.f90, %.mod, $(src_SOURCES))


# Create dependency tree such that we may bypass phony targets
netcdf_ncdf.o: netcdf_ncdf.f90
netcdf_ncdf.f90: netcdf_ncdf_pp.F90 netcdf_ncdf_att_inc.inc netcdf_ncdf_var_inc.inc ncdf.sh
	VPATH="$(TOP_DIR)/src" $(src_DIR)/ncdf.sh
	$(FPP) $(FPPFLAGS) $(INCLUDES) -I$(src_DIR) -I. $< | sed -f $(src_DIR)/filter.sed > $@


# Define objects that needs compilation
SOURCES += $(src_SOURCES)
FPP_SOURCES += $(src_FPP_SOURCES)
F_MODS  += $(src_MODS)
OBJECTS += $(src_OBJECTS)


# Ensure that the objects are built with the correct module path
$(src_FPP_SOURCES): INCLUDES += -I$(src_DIR)

.PHONY: source-src
source-src: $(src_SOURCES)
	mkdir -p $(SOURCES_DIR)/src
	mv $(src_FPP_SOURCES) $(SOURCES_DIR)/src


.PHONY: clean-src
clean-src:
	-rm -f $(src_FPP_SOURCES)
	-rm -f $(src_OBJECTS)
	-rm -f *.mod
	-rm -f netcdf_ncdf_funcs.inc netcdf_ncdf_interface.inc

clean: clean-src


# Local Variables:
#  mode: makefile-gmake
# End:
