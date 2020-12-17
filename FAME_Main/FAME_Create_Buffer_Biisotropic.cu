#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include <assert.h>

int FAME_Create_Buffer_Biisotropic(	CULIB_HANDLES*  cuHandles, 
						FFT_BUFFER* 	fft_buffer,
						LANCZOS_BUFFER* lBuffer,
						int N, int Nstep)
{
    int N3 = N * 3;
	size_t memsize;

    memsize = N3 * sizeof(cufftDoubleComplex);
    checkCudaErrors(cudaMalloc((void**)&fft_buffer->d_A, 2*memsize));
    checkCudaErrors(cudaMalloc((void**)&fft_buffer->dvec_x, memsize));
    checkCudaErrors(cudaMalloc((void**)&fft_buffer->tmp, memsize));

    memsize = N3 * sizeof(cuDoubleComplex);
    checkCudaErrors(cudaMalloc((void**)&cuHandles->N3_temp1, memsize));
    checkCudaErrors(cudaMalloc((void**)&cuHandles->N3_temp2, memsize));

    memsize = Nstep * Nstep * sizeof(cuDoubleComplex);
    checkCudaErrors(cudaMalloc((void**) &lBuffer->dz, memsize));

	memsize = Nstep * Nstep * sizeof(cmpx);
	lBuffer->z   = (cmpx*) malloc(memsize);   assert(lBuffer->z != NULL);

	memsize = Nstep * sizeof(double);
    lBuffer->T0  = (double*) malloc(memsize); assert(lBuffer->T0 != NULL);
    lBuffer->LT0 = (double*) malloc(memsize); assert(lBuffer->LT0 != NULL);
    lBuffer->T2  = (double*) malloc(memsize); assert(lBuffer->T2 != NULL);
    lBuffer->LT1 = (double*) malloc(memsize); assert(lBuffer->LT1 != NULL);

    memsize = (Nstep+1) * sizeof(double);
    lBuffer->T1  = (double*) malloc(memsize); assert(lBuffer->T1 != NULL); 

	return 0;
}

