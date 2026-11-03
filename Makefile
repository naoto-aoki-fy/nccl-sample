.DEFAULT_GOAL := all

# Cleanup needs no toolchain configuration; every build still requires it.
ifeq ($(strip $(MAKECMDGOALS)),clean)
-include config.mk
else
include config.mk
endif

NVCC ?= nvcc
NVCCFLAGS ?= -O2 -std=c++17
NCCL_HOME ?= /opt/nvidia/hpc_sdk/Linux_aarch64/26.9/comm_libs/13.3/nccl
NCCL_LIBDIR ?= $(if $(wildcard $(NCCL_HOME)/lib64),$(NCCL_HOME)/lib64,$(NCCL_HOME)/lib)

.PHONY: all clean
all: nccl_allreduce

allreduce.o: allreduce.cu Makefile config.mk
	$(NVCC) --forward-unknown-to-host-compiler $(CPPFLAGS) $(NVCCFLAGS) $(CFLAGS_VENDOR) $(GENCODE_FLAGS) -I"$(NCCL_HOME)/include" -c $< -o $@

nccl_allreduce: allreduce.o
	$(NVCC) --forward-unknown-to-host-compiler $(NVCCFLAGS) $(GENCODE_FLAGS) $^ -o $@ $(LDFLAGS) $(LDFLAGS_VENDOR) -L"$(NCCL_LIBDIR)" -Xlinker -rpath -Xlinker "$(NCCL_LIBDIR)" -lnccl

clean:
	rm -f allreduce.o nccl_allreduce
