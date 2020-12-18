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

    memsize = N3 * sizeof(cmpxfft);
    checkCudaErrors(cudaMalloc((void**)&fft_buffer->d_A, 2*memsize));
    checkCudaErrors(cudaMalloc((void**)&fft_buffer->dvec_x, memsize));
    checkCudaErrors(cudaMalloc((void**)&fft_buffer->tmp, memsize));

    memsize = N3 * sizeof(cmpxGPU);
    checkCudaErrors(cudaMalloc((void**)&cuHandles->N3_temp1, memsize));
    checkCudaErrors(cudaMalloc((void**)&cuHandles->N3_temp2, memsize));

    memsize = Nstep * Nstep * sizeof(cmpxGPU);
    checkCudaErrors(cudaMalloc((void**) &lBuffer->dz, memsize));

	memsize = Nstep * Nstep * sizeof(cmpxCPU);
	lBuffer->z   = (cmpxCPU*) malloc(memsize);   assert(lBuffer->z != NULL);

	memsize = Nstep * sizeof(realCPU);
    lBuffer->T0  = (realCPU*) malloc(memsize); assert(lBuffer->T0 != NULL);
    lBuffer->LT0 = (realCPU*) malloc(memsize); assert(lBuffer->LT0 != NULL);
    lBuffer->T2  = (realCPU*) malloc(memsize); assert(lBuffer->T2 != NULL);
    lBuffer->LT1 = (realCPU*) malloc(memsize); assert(lBuffer->LT1 != NULL);

    memsize = (Nstep+1) * sizeof(realCPU);
    lBuffer->T1  = (realCPU*) malloc(memsize); assert(lBuffer->T1 != NULL); 

	return 0;
}

