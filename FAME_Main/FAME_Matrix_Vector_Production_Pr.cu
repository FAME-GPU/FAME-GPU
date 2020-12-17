#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include "FAME_FFT_CUDA.cuh"
#include "printDeviceArray.cuh"

static __global__ void vp_add_vp(int size, cuDoubleComplex* L_1, cuDoubleComplex* L_2, cuDoubleComplex* vec_1, cuDoubleComplex* vec_2,cuDoubleComplex* vec_out);

////////////=========================== Create Pr function for Biiso (cuda)===========================//////////////////
int FAME_Matrix_Vector_Production_Pr(   CULIB_HANDLES cuHandles, 
                                        FFT_BUFFER fft_buffer, 
                                        cuDoubleComplex* vec_x, 
                                        int Nx, int Ny, int Nz, int Nd, 
                                        cuDoubleComplex* D_k,
                                        cuDoubleComplex* Pi_Pr, 
                                        cuDoubleComplex* vec_y)
{
    int N = Nx*Ny*Nz;
    //cuDoubleComplex* temp = cuHandles.N3_temp1;
	cuDoubleComplex* temp;
    checkCudaErrors(cudaMalloc((void**)&temp, 3*N*sizeof(cuDoubleComplex)));
    dim3 DimBlock(BLOCK_SIZE,1,1);
    dim3 DimGrid((Nd-1)/BLOCK_SIZE +1,1,1);

	// Initial

    checkCudaErrors(cudaMemset(temp, 0, N * 3 * sizeof(cuDoubleComplex)));

    vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Pr,         Pi_Pr+3*Nd, vec_x, vec_x+Nd, temp+N-Nd);
    cudaDeviceSynchronize();
    vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Pr+Nd,      Pi_Pr+4*Nd, vec_x, vec_x+Nd, temp+N-Nd+N);
    cudaDeviceSynchronize();
    vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Pr+2*Nd,    Pi_Pr+5*Nd, vec_x, vec_x+Nd, temp+N-Nd+2*N);
    cudaDeviceSynchronize();

  
    IFFT_CUDA(vec_y, temp, D_k, fft_buffer, cuHandles, Nx, Ny, Nz);

	cudaFree(temp);

    return 0;
}

int FAME_Matrix_Vector_Production_Pr(CULIB_HANDLES cuHandles, FFT_BUFFER fft_buffer, cuDoubleComplex* vec_x, int Nx, int Ny, int Nz, int Nd, cuDoubleComplex* D_kx, cuDoubleComplex* D_ky, cuDoubleComplex* D_kz, cuDoubleComplex* Pi_Pr, cuDoubleComplex* vec_y)
{
    int N = Nx*Ny*Nz;
    int N3 = N * 3;
    dim3 DimBlock(BLOCK_SIZE,1,1);
    dim3 DimGrid((Nd-1)/BLOCK_SIZE +1,1,1);
    cuDoubleComplex* temp;
    checkCudaErrors(cudaMalloc((void**)&temp, N3*sizeof(cuDoubleComplex)));

    checkCudaErrors(cudaMemset(temp, 0, N3 * sizeof(cuDoubleComplex)));

    //printDeviceArray( vec_x, 2*Nd, "print_vec_x.txt");
    vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Pr,         Pi_Pr+3*Nd, vec_x, vec_x+Nd, temp+N-Nd);
    vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Pr+Nd,      Pi_Pr+4*Nd, vec_x, vec_x+Nd, temp+N-Nd+N);
    vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Pr+2*Nd,    Pi_Pr+5*Nd, vec_x, vec_x+Nd, temp+N-Nd+2*N);
    
    //printDeviceArray( temp, 3*N, "print_temp.txt");
	for(int i=0; i<3; i++)
        spMV_fastT_gpu( vec_y+i*N, temp+i*N, cuHandles, &fft_buffer, D_kx, D_ky, D_kz, Nx, Ny, Nz, 1);

    cudaFree(temp);
    return 0;
}

static __global__ void vp_add_vp(int size, cuDoubleComplex* L_1, cuDoubleComplex* L_2, cuDoubleComplex* vec_1, cuDoubleComplex* vec_2,cuDoubleComplex* vec_out)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < size)
    {
        //vec_out[idx] = L_1[idx]*vec_1[idx] + L_2[idx]*vec_2[idx];
        vec_out[idx].x = L_1[idx].x*vec_1[idx].x + L_2[idx].x*vec_2[idx].x - L_1[idx].y*vec_1[idx].y - L_2[idx].y*vec_2[idx].y;
        vec_out[idx].y = L_1[idx].x*vec_1[idx].y + L_2[idx].y*vec_2[idx].x + L_1[idx].y*vec_1[idx].x + L_2[idx].x*vec_2[idx].y;

    }

}