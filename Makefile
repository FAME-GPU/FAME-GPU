#############################################
### Project : FAME ( Isotropic GPU )      ###
### Created by Sheng Wang.                ###
#############################################
#############################################
### User need to check                    ###
### GPU_TARGET CUDADIR LAPACKDIR          ###
#############################################
# Fermi   - NVIDIA compute capability 2.x cards
# Kepler  - NVIDIA compute capability 3.x cards
# Maxwell - NVIDIA compute capability 5.x cards
# Pascal  - NVIDIA compute capability 6.x cards
# Volta   - NVIDIA compute capability 7.x cards
# Note that NVIDIA no longer supports 1.x cards
GPU_TARGET = Pascal
include make.inc
#############################################
### Compiler settings                     ###
#############################################
CC      = g++
NVCC    = nvcc
CCFLA   = -O3 -m64
NVCCFLA = -O3 -m64 $(NVCCFLAGS) -std=c++03
#############################################
### CUDA                                  ###
#############################################
CUDADIR ?= /usr/local/cuda-10.1
CUDAINC  = -I$(CUDADIR)/include -I$(CUDADIR)/samples/common/inc
CUDALIB  = -L$(CUDADIR)/lib64
CUDAFLA  = -lcudart -lcublas -lcufft -lcusolver -lcusparse -lcurand -lpthread -ldl -lgomp
CUDAALL  = $(CUDAINC) $(CUDALIB) $(CUDAFLA)
#############################################
### LAPACK                                ###
#############################################
LAPACKDIR ?= /home/zgxt/softwares/lapack-3.9.0
LAPACKINC  = -I$(LAPACKDIR)/LAPACKE/include
LAPACKLIB  = -L$(LAPACKDIR)
LAPACKFLA  = -llapacke -llapack -lgfortran -lrefblas
LAPACKALL  = $(LAPACKINC) $(LAPACKLIB) $(LAPACKFLA)
#############################################
MAIDIR = ./FAME_Main
PREDIR = ./FAME_Preprocessing
TOODIR = ./FAME_Tools
OBJDIR = ./obj
INCDIR = ./include
SRCCPP = $(wildcard $(MAIDIR)/*.cpp) \
		 $(wildcard $(PREDIR)/*.cpp) \
		 $(wildcard $(TOODIR)/*.cpp) 
SRCCU  = $(wildcard $(MAIDIR)/*.cu) \
		 $(wildcard $(PREDIR)/*.cu) \
		 $(wildcard $(TOODIR)/*.cu)
OBJCPP = $(patsubst %.cpp, %.o, $(SRCCPP))
OBJCU  = $(patsubst %.cu, %.o, $(SRCCU))
OBJALL = $(OBJDIR)/Main.o $(patsubst %.cpp, $(OBJDIR)/%.o, $(notdir $(SRCCPP))) \
						  $(patsubst %.cu,  $(OBJDIR)/%.o, $(notdir $(SRCCU)))
#############################################
.PHONY: all clean

all: MakeExe

MakeExe: Main.out

Main.out: Main.o $(OBJCPP) $(OBJCU)
	@$(CC) -o $@ $(OBJALL) $(CCFLA) -I$(INCDIR) $(CUDAALL) $(LAPACKALL) 

%.o:%.cpp
	@echo "$(CC) $<"
	@$(CC) -c $< -o $(OBJDIR)/$@ $(CCFLA) -I$(INCDIR)

$(OBJCPP):%.o:%.cpp
	@echo "$(CC) $<"
	@$(CC) -c $< -o $(OBJDIR)/$(notdir $@) $(CCFLA) -I$(INCDIR)

$(OBJCU): %.o:%.cu
	@echo "$(NVCC) $<"
	@$(NVCC) -c $< -o $(OBJDIR)/$(notdir $@) $(NVCCFLA) -I$(INCDIR) $(CUDAALL) $(LAPACKALL) 

clean:
	@-rm Main.out
	@-rm -f $(OBJALL)
	@-rm -f Data*
	@-rm -f print_*
