# cuGMEC build
#
#   make                     builds ./cuGMEC from src/cuGMEC_param.h
#   make CASE=runs/itpa      builds runs/itpa/cuGMEC from runs/itpa/cuGMEC_param.h

NVCC       ?= nvcc
ARCH       ?= sm_80
CUDSS_HOME ?=
NCCL_HOME  ?=
CASE       ?=

SRC   := src/cuGMEC_main.cu
PARAM := $(abspath $(if $(CASE),$(CASE)/cuGMEC_param.h,src/cuGMEC_param.h))
BUILD := $(if $(CASE),$(CASE)/build,build)
BIN   := $(if $(CASE),$(CASE)/cuGMEC,cuGMEC)

# The parameter file is compiled in and its text is embedded, so the run records exactly what it was built with.
SNAPSHOT := $(BUILD)/cuGMEC_param_snapshot.inc

INC_FLAGS := -Isrc -I$(BUILD) \
             $(if $(CUDSS_HOME),-I$(CUDSS_HOME)/include) \
             $(if $(NCCL_HOME),-I$(NCCL_HOME)/include)
LIB_FLAGS := $(if $(CUDSS_HOME),-L$(CUDSS_HOME)/lib) \
             $(if $(NCCL_HOME),-L$(NCCL_HOME)/lib)

LIBS      := -lmpi -lnccl -lcufft -lcudss -lcusparse
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
