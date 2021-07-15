#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include <complex.h>
#include "printDeviceArray.cuh"


/*
Sigma = blkdiag(Lambda_q_sqrt, Lambda_q_sqrt);
*/

static __global__ void dot_prod( int size, realGPU* Sigma, cmpxGPU* vec, cmpxGPU* result);
static __global__ void dot_prod0( int size, realGPU* Sigma, cmpxGPU* vec, cmpxGPU* result);



void FAME_Matrix_Vector_Production_Bianisotropic_i_invSigma(CULIB_HANDLES cuHandles, int Nd, realGPU* Lambda_q_sqrt, cmpxGPU* x, cmpxGPU *y)
{

    int Nd2 = Nd * 2;
    dim3 DimBlock( BLOCK_SIZE, 1, 1);
    dim3 DimGrid( (Nd2-1)/BLOCK_SIZE +1, 1, 1);

    dot_prod<<<DimGrid, DimBlock>>>(Nd2, Lambda_q_sqrt, x+24*Nd, y);
    dot_prod0<<<DimGrid, DimBlock>>>(Nd2, Lambda_q_sqrt, x+16*Nd, y+8*Nd);

    dot_prod<<<DimGrid, DimBlock>>>(Nd2, Lambda_q_sqrt, x+8*Nd, y+16*Nd);
    dot_prod0<<<DimGrid, DimBlock>>>(Nd2, Lambda_q_sqrt, x, y+24*Nd);

}


static __global__ void dot_prod( int size, realGPU* Sigma, cmpxGPU* vec, cmpxGPU* result)
{
    int i;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < size )
    {
        for( i = 0; i < 4; i++)
        {
            result[idx + i * size].x = - vec[idx + i * size].y / Sigma[idx];
            result[idx + i * size].y = vec[idx + i * size].x / Sigma[idx];
        }
        
    }

}

static __global__ void dot_prod0( int size, realGPU* Sigma, cmpxGPU* vec, cmpxGPU* result)
{
    int i;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < size )
    {
        for( i = 0; i < 4; i++)
        {
            result[idx + i * size].x = vec[idx + i * size].y / Sigma[idx];
            result[idx + i * size].y = - vec[idx + i * size].x / Sigma[idx];
        }
        
    }

}