# cuGMEC build
#
#   make                     builds ./cuGMEC from src/cuGMEC_param.h
#   make CASE=runs/itpa      builds runs/itpa/cuGMEC from runs/itpa/cuGMEC_param.h
#
# MPI comes from the compiler's default paths (-lmpi) or from MPI_HOME. On a Cray system (Perlmutter, with the
# cudatoolkit, cray-mpich and nccl modules) it defaults to CRAY_MPICH_DIR with -lmpich, and the CUDA math libraries
# (cuFFT, cuSPARSE) are taken from the cudatoolkit module.

NVCC       ?= nvcc
ARCH       ?= sm_80
CUDSS_HOME ?=
NCCL_HOME  ?=
CASE       ?=

ifdef CRAY_MPICH_DIR
MPI_HOME   ?= $(CRAY_MPICH_DIR)
MPI_LIBS   ?= -lmpich
endif
MPI_HOME   ?=
MPI_LIBS   ?= -lmpi
comma      := ,

SRC   := src/cuGMEC_main.cu
PARAM := $(abspath $(if $(CASE),$(CASE)/cuGMEC_param.h,src/cuGMEC_param.h))
BUILD := $(if $(CASE),$(CASE)/build,build)
BIN   := $(if $(CASE),$(CASE)/cuGMEC,cuGMEC)

# The parameter file is compiled in and its text is embedded, so the run records exactly what it was built with.
SNAPSHOT := $(BUILD)/cuGMEC_param_snapshot.inc

INC_FLAGS := -Isrc -I$(BUILD) \
             $(foreach d,$(CUDSS_HOME) $(NCCL_HOME) $(MPI_HOME),-I$(d)/include) \
             $(filter -I%,$(CRAY_CUDATOOLKIT_INCLUDE_OPTS))
LIB_FLAGS := $(foreach d,$(CUDSS_HOME) $(NCCL_HOME) $(MPI_HOME),-L$(d)/lib -Xlinker -rpath$(comma)$(d)/lib) \
             $(filter -L%,$(CRAY_CUDATOOLKIT_POST_LINK_OPTS))

LIBS      := $(MPI_LIBS) -lnccl -lcufft -lcudss -lcusparse
NVCCFLAGS := -arch=$(ARCH) -std=c++20 --expt-relaxed-constexpr -Xcompiler -fopenmp -DCUGMEC_PARAM_H='"$(PARAM)"'

DEPS := $(wildcard src/*.h) $(wildcard src/kernels/*.cuh) $(PARAM)

.PHONY: all clean

all: $(BIN)

$(SNAPSHOT): $(PARAM)
	@mkdir -p $(BUILD)
	{ printf 'R"cugmec_param('; cat $<; printf ')cugmec_param"\n'; } > $@

$(BIN): $(SRC) $(DEPS) $(SNAPSHOT)
	$(NVCC) $(INC_FLAGS) $(LIB_FLAGS) $(NVCCFLAGS) $(SRC) $(LIBS) -o $(BIN)

clean:
	rm -f $(BIN) $(SNAPSHOT)
