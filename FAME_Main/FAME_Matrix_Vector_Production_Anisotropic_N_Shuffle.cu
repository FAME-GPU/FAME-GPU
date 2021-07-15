#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include "printDeviceArray.cuh"


static __global__ void N_vec_prod(int N, cmpxGPU* N_in, 
    cmpxGPU* tmp_vec, cmpxGPU* vec, 
    int* InOut_index, int len1, int len2, int len3, int len4);

int FAME_Matrix_Vector_Production_Anisotropic_N_Shuffle(CULIB_HANDLES cuHandles, int Nx, int Ny, int Nz, int Nd, 
    int *InOut_index, int *InOut_index_length, cmpxGPU* N_in, cmpxGPU* vec)
{
    int N = Nx * Ny * Nz;
    cmpxGPU* temp_vec = cuHandles.temp_vec2;
    cudaMemcpy(temp_vec, vec, N*12*sizeof(cmpxGPU), cudaMemcpyDeviceToDevice);

    dim3 DimBlock(BLOCK_SIZE, 1, 1);
    dim3 DimGrid((InOut_index_length[3] - 1) / BLOCK_SIZE + 1, 1, 1); 

    N_vec_prod<<<DimGrid, DimBlock>>>(N, N_in, temp_vec, vec, InOut_index, 
        InOut_index_length[0], InOut_index_length[1], InOut_index_length[2], InOut_index_length[3]);
    cudaDeviceSynchronize();

    return 0;
}

static __global__ void N_vec_prod(int N, cmpxGPU* N_in, 
    cmpxGPU* temp_vec, cmpxGPU* vec, 
    int* InOut_index, int len1, int len2, int len3, int len4)
{  

    int InOut_idx;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if( idx < len1 )
    {     
        InOut_idx = InOut_index[idx];

        vec[InOut_idx].x        = N_in[0].x * temp_vec[InOut_idx].x + N_in[3].x * temp_vec[InOut_idx + 4*N].x + N_in[6].x * temp_vec[InOut_idx + 8*N].x;
        vec[InOut_idx].y        = N_in[0].x * temp_vec[InOut_idx].y + N_in[3].x * temp_vec[InOut_idx + 4*N].y + N_in[6].x * temp_vec[InOut_idx + 8*N].y;

        vec[InOut_idx + 4*N].x  = N_in[1].x * temp_vec[InOut_idx].x + N_in[4].x * temp_vec[InOut_idx + 4*N].x + N_in[7].x * temp_vec[InOut_idx + 8*N].x;
        vec[InOut_idx + 4*N].y  = N_in[1].x * temp_vec[InOut_idx].y + N_in[4].x * temp_vec[InOut_idx + 4*N].y + N_in[7].x * temp_vec[InOut_idx + 8*N].y;

        vec[InOut_idx + 8*N].x  = N_in[2].x * temp_vec[InOut_idx].x + N_in[5].x * temp_vec[InOut_idx + 4*N].x + N_in[8].x * temp_vec[InOut_idx + 8*N].x;
        vec[InOut_idx + 8*N].y  = N_in[2].x * temp_vec[InOut_idx].y + N_in[5].x * temp_vec[InOut_idx + 4*N].y + N_in[8].x * temp_vec[InOut_idx + 8*N].y;

    }

    else if( idx >= len1 && idx < len2 )
    {     
        InOut_idx = InOut_index[idx];

        vec[InOut_idx + 3*N].x  = N_in[0].x * temp_vec[InOut_idx + 3*N].x + N_in[3].x * temp_vec[InOut_idx + 1*N].x + N_in[6].x * temp_vec[InOut_idx + 11*N].x;
        vec[InOut_idx + 3*N].y  = N_in[0].x * temp_vec[InOut_idx + 3*N].y + N_in[3].x * temp_vec[InOut_idx + 1*N].y + N_in[6].x * temp_vec[InOut_idx + 11*N].y;

        vec[InOut_idx + 1*N].x  = N_in[1].x * temp_vec[InOut_idx + 3*N].x + N_in[4].x * temp_vec[InOut_idx + 1*N].x + N_in[7].x * temp_vec[InOut_idx + 11*N].x;
        vec[InOut_idx + 1*N].y  = N_in[1].x * temp_vec[InOut_idx + 3*N].y + N_in[4].x * temp_vec[InOut_idx + 1*N].y + N_in[7].x * temp_vec[InOut_idx + 11*N].y;

        vec[InOut_idx + 11*N].x  = N_in[2].x * temp_vec[InOut_idx + 3*N].x + N_in[5].x * temp_vec[InOut_idx + 1*N].x + N_in[8].x * temp_vec[InOut_idx + 11*N].x;
        vec[InOut_idx + 11*N].y  = N_in[2].x * temp_vec[InOut_idx + 3*N].y + N_in[5].x * temp_vec[InOut_idx + 1*N].y + N_in[8].x * temp_vec[InOut_idx + 11*N].y;

    }

    else if( idx >= len2 && idx < len3 )
    {     
        InOut_idx = InOut_index[idx];

        vec[InOut_idx + 6*N].x  = N_in[0].x * temp_vec[InOut_idx + 6*N].x + N_in[3].x * temp_vec[InOut_idx + 10*N].x + N_in[6].x * temp_vec[InOut_idx + 2*N].x;
        vec[InOut_idx + 6*N].y  = N_in[0].x * temp_vec[InOut_idx + 6*N].y + N_in[3].x * temp_vec[InOut_idx + 10*N].y + N_in[6].x * temp_vec[InOut_idx + 2*N].y;

        vec[InOut_idx + 10*N].x  = N_in[1].x * temp_vec[InOut_idx + 6*N].x + N_in[4].x * temp_vec[InOut_idx + 10*N].x + N_in[7].x * temp_vec[InOut_idx + 2*N].x;
        vec[InOut_idx + 10*N].y  = N_in[1].x * temp_vec[InOut_idx + 6*N].y + N_in[4].x * temp_vec[InOut_idx + 10*N].y + N_in[7].x * temp_vec[InOut_idx + 2*N].y;

        vec[InOut_idx + 2*N].x  = N_in[2].x * temp_vec[InOut_idx + 6*N].x + N_in[5].x * temp_vec[InOut_idx + 10*N].x + N_in[8].x * temp_vec[InOut_idx + 2*N].x;
        vec[InOut_idx + 2*N].y  = N_in[2].x * temp_vec[InOut_idx + 6*N].y + N_in[5].x * temp_vec[InOut_idx + 10*N].y + N_in[8].x * temp_vec[InOut_idx + 2*N].y;

    }

    else if( idx >= len3 && idx < len4 )
    {     
        InOut_idx = InOut_index[idx];

        vec[InOut_idx + 9*N].x  = N_in[0].x * temp_vec[InOut_idx + 9*N].x + N_in[3].x * temp_vec[InOut_idx + 7*N].x + N_in[6].x * temp_vec[InOut_idx + 5*N].x;
        vec[InOut_idx + 9*N].y  = N_in[0].x * temp_vec[InOut_idx + 9*N].y + N_in[3].x * temp_vec[InOut_idx + 7*N].y + N_in[6].x * temp_vec[InOut_idx + 5*N].y;

        vec[InOut_idx + 7*N].x  = N_in[1].x * temp_vec[InOut_idx + 9*N].x + N_in[4].x * temp_vec[InOut_idx + 7*N].x + N_in[7].x * temp_vec[InOut_idx + 5*N].x;
        vec[InOut_idx + 7*N].y  = N_in[1].x * temp_vec[InOut_idx + 9*N].y + N_in[4].x * temp_vec[InOut_idx + 7*N].y + N_in[7].x * temp_vec[InOut_idx + 5*N].y;

        vec[InOut_idx + 5*N].x  = N_in[2].x * temp_vec[InOut_idx + 9*N].x + N_in[5].x * temp_vec[InOut_idx + 7*N].x + N_in[8].x * temp_vec[InOut_idx + 5*N].x;
        vec[InOut_idx + 5*N].y  = N_in[2].x * temp_vec[InOut_idx + 9*N].y + N_in[5].x * temp_vec[InOut_idx + 7*N].y + N_in[8].x * temp_vec[InOut_idx + 5*N].y;

    }
}

/*
static __global__ void N_vec_prod(cmpxGPU* N_in, 
    cmpxGPU* vec0, cmpxGPU* vec1, cmpxGPU* vec2,
    cmpxGPU* temp_vec0, cmpxGPU* temp_vec1, cmpxGPU* temp_vec2, 
    int* InOut_index, int InOut_index_length);

int FAME_Matrix_Vector_Production_Anisotropic_N_Shuffle(CULIB_HANDLES cuHandles, int Nx, int Ny, int Nz, int Nd, 
    int *InOut_index, int *InOut_index_length, cmpxGPU* N_in, cmpxGPU* vec)
{
    int N = Nx * Ny * Nz;
    cmpxGPU* temp_vec = cuHandles.temp_vec2;
    cudaMemcpy(temp_vec, vec, N*12*sizeof(cmpxGPU), cudaMemcpyDeviceToDevice);

    dim3 DimBlock(BLOCK_SIZE, 1, 1);
    dim3 DimGrid((N - 1) / BLOCK_SIZE + 1, 1, 1); 

    N_vec_prod<<<DimGrid, DimBlock>>>(N_in, temp_vec, temp_vec+4*N, temp_vec+8*N, vec, vec+4*N,  vec+8*N, 
        InOut_index, InOut_index_length[0]);
    cudaDeviceSynchronize();
	N_vec_prod<<<DimGrid, DimBlock>>>(N_in, temp_vec+3*N, temp_vec+1*N, temp_vec+11*N, vec+3*N, vec+1*N,  vec+11*N, 
        InOut_index + InOut_index_length[0], InOut_index_length[1] - InOut_index_length[0]);
    cudaDeviceSynchronize();
	N_vec_prod<<<DimGrid, DimBlock>>>(N_in, temp_vec+6*N, temp_vec+10*N, temp_vec+2*N, vec+6*N, vec+10*N,  vec+2*N, 
        InOut_index + InOut_index_length[1], InOut_index_length[2] - InOut_index_length[1]);
    cudaDeviceSynchronize();
	N_vec_prod<<<DimGrid, DimBlock>>>(N_in, temp_vec+9*N, temp_vec+7*N, temp_vec+5*N, vec+9*N, vec+7*N,  vec+5*N, 
        InOut_index + InOut_index_length[2], InOut_index_length[3] - InOut_index_length[2]);
    cudaDeviceSynchronize();

    return 0;
}

static __global__ void N_vec_prod(cmpxGPU* N_in, 
    cmpxGPU* vec0, cmpxGPU* vec1, cmpxGPU* vec2,
    cmpxGPU* temp_vec0, cmpxGPU* temp_vec1, cmpxGPU* temp_vec2, 
    int* InOut_index, int InOut_index_length)
{  

    int InOut_idx;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if( idx < InOut_index_length )
    {     
        InOut_idx = InOut_index[idx];

        temp_vec0[InOut_idx].x  = N_in[0].x * vec0[InOut_idx].x + N_in[3].x * vec1[InOut_idx].x + N_in[6].x * vec2[InOut_idx].x;
        temp_vec0[InOut_idx].y  = N_in[0].x * vec0[InOut_idx].y + N_in[3].x * vec1[InOut_idx].y + N_in[6].x * vec2[InOut_idx].y;

        temp_vec1[InOut_idx].x  = N_in[1].x * vec0[InOut_idx].x + N_in[4].x * vec1[InOut_idx].x + N_in[7].x * vec2[InOut_idx].x;
        temp_vec1[InOut_idx].y  = N_in[1].x * vec0[InOut_idx].y + N_in[4].x * vec1[InOut_idx].y + N_in[7].x * vec2[InOut_idx].y;

        temp_vec2[InOut_idx].x  = N_in[2].x * vec0[InOut_idx].x + N_in[5].x * vec1[InOut_idx].x + N_in[8].x * vec2[InOut_idx].x;
        temp_vec2[InOut_idx].y  = N_in[2].x * vec0[InOut_idx].y + N_in[5].x * vec1[InOut_idx].y + N_in[8].x * vec2[InOut_idx].y;

    }
}
*/