#############################################
### NVCC FLAG Setting                     ###
#############################################
# NVCC options for the different cards
ifneq ($(findstring Kepler, $(GPU_TARGET)),)
    GPU_TARGET += sm_30 sm_35
endif
ifneq ($(findstring Maxwell, $(GPU_TARGET)),)
    GPU_TARGET += sm_50
endif
ifneq ($(findstring Pascal, $(GPU_TARGET)),)
    GPU_TARGET += sm_60
endif
ifneq ($(findstring Volta, $(GPU_TARGET)),)
    GPU_TARGET += sm_70
endif
ifneq ($(findstring Turing, $(GPU_TARGET)),)
    GPU_TARGET += sm_75
endif

NV_SM    :=
NV_COMP  :=

ifneq ($(findstring sm_10, $(GPU_TARGET)),)
    $(warning CUDA arch 1.x is no longer supported by CUDA >= 6.x and MAGMA >= 2.0)
endif
ifneq ($(findstring sm_13, $(GPU_TARGET)),)
    $(warning CUDA arch 1.x is no longer supported by CUDA >= 6.x and MAGMA >= 2.0)
endif
ifneq ($(findstring sm_20, $(GPU_TARGET)),)
    NV_SM    += -gencode arch=compute_20,code=sm_20
    NV_COMP  := -gencode arch=compute_20,code=compute_20
    $(warning CUDA arch 2.x is no longer supported by CUDA >= 9.x)
endif
ifneq ($(findstring sm_30, $(GPU_TARGET)),)
    NV_SM    += -gencode arch=compute_30,code=sm_30
    NV_COMP  := -gencode arch=compute_30,code=compute_30
endif
ifneq ($(findstring sm_32, $(GPU_TARGET)),)
    NV_SM    += -gencode arch=compute_32,code=sm_32
    NV_COMP  := -gencode arch=compute_32,code=compute_32
endif
ifneq ($(findstring sm_35, $(GPU_TARGET)),)
    NV_SM    += -gencode arch=compute_35,code=sm_35
    NV_COMP  := -gencode arch=compute_35,code=compute_35
endif
ifneq ($(findstring sm_50, $(GPU_TARGET)),)
    NV_SM    += -gencode arch=compute_50,code=sm_50
    NV_COMP  := -gencode arch=compute_50,code=compute_50
endif
ifneq ($(findstring sm_52, $(GPU_TARGET)),)
    MIN_ARCH ?= 520
    NV_SM    += -gencode arch=compute_52,code=sm_52
    NV_COMP  := -gencode arch=compute_52,code=compute_52
endif
ifneq ($(findstring sm_53, $(GPU_TARGET)),)
    NV_SM    += -gencode arch=compute_53,code=sm_53
    NV_COMP  := -gencode arch=compute_53,code=compute_53
endif
ifneq ($(findstring sm_60, $(GPU_TARGET)),)
    NV_SM    += -gencode arch=compute_60,code=sm_60
    NV_COMP  := -gencode arch=compute_60,code=compute_60
endif
ifneq ($(findstring sm_61, $(GPU_TARGET)),)
    NV_SM    += -gencode arch=compute_61,code=sm_61
    NV_COMP  := -gencode arch=compute_61,code=compute_61
endif
ifneq ($(findstring sm_62, $(GPU_TARGET)),)
    NV_SM    += -gencode arch=compute_62,code=sm_62
    NV_COMP  := -gencode arch=compute_62,code=compute_62
endif
ifneq ($(findstring sm_70, $(GPU_TARGET)),)
    NV_SM    += -gencode arch=compute_70,code=sm_70
    NV_COMP  := -gencode arch=compute_70,code=compute_70
endif
ifneq ($(findstring sm_71, $(GPU_TARGET)),)
    NV_SM    += -gencode arch=compute_71,code=sm_71
    NV_COMP  := -gencode arch=compute_71,code=compute_71
endif
ifneq ($(findstring sm_75, $(GPU_TARGET)),)
    NV_SM    += -gencode arch=compute_75,code=sm_75
    NV_COMP  := -gencode arch=compute_75,code=compute_75
endif
ifeq ($(NV_COMP),)
    $(error GPU_TARGET, currently $(GPU_TARGET), must contain one or more of Fermi, Kepler, Maxwell, Pascal, Volta, Turing. Please edit Makefile)
endif

NVCCFLAGS += $(NV_SM) $(NV_COMP)