#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include "FAME_FFT_CUDA.cuh"
#include "printDeviceArray.cuh"
static __global__ void vp_add_vp_add_vp(int N, int Nd, int Nd_2, cmpxGPU* L, cmpxGPU* vec, cmpxGPU* vec_out);


////////////=========================== Create Prs function for Biiso (cuda)===========================//////////////////SC
int FAME_Matrix_Vector_Production_Prs( CULIB_HANDLES cuHandles, 
                                                                                        FFT_BUFFER fft_buffer, 
                                                                                        cmpxGPU* vec_x, 
                                                                                        int Nx, int Ny, int Nz, int Nd,
                                                                                        cmpxGPU* D_ks, 
                                                                                        cmpxGPU* Pi_Prs, 
                                                                                         cmpxGPU* vec_y)
{
    int N = Nx*Ny*Nz;

	cmpxGPU* tmp;
    cudaMalloc((void**)&tmp, 3*N*sizeof(cmpxGPU));
    dim3 DimBlock(BLOCK_SIZE,1,1);
    dim3 DimGrid((Nd-1)/BLOCK_SIZE +1,1,1);

    FFT_CUDA(tmp, vec_x, D_ks, fft_buffer, cuHandles, Nx, Ny, Nz);
	
    vp_add_vp_add_vp<<<DimGrid, DimBlock>>>(N, Nd, 2*Nd, Pi_Prs, tmp+(N-Nd), vec_y);
    vp_add_vp_add_vp<<<DimGrid, DimBlock>>>(N, Nd, 2*Nd, Pi_Prs+Nd, tmp+(N-Nd), vec_y+Nd);
 
    
    cudaFree(tmp);

    return 0;
}

int FAME_Matrix_Vector_Production_Prs( CULIB_HANDLES cuHandles, FFT_BUFFER fft_buffer, cmpxGPU* vec_x, int Nx, int Ny, int Nz, int Nd, cmpxGPU* D_kx, cmpxGPU* D_ky, cmpxGPU* D_kz, cmpxGPU* Pi_Prs, cmpxGPU* vec_y)//FCC
{
    int N = Nx*Ny*Nz;
    cmpxGPU* tmp;
    cudaMalloc((void**)&tmp, 3*N*sizeof(cmpxGPU));

    dim3 DimBlock(BLOCK_SIZE,1,1);
    dim3 DimGrid((Nd-1)/BLOCK_SIZE +1,1,1);

	for(int i=0; i<3; i++)
        spMV_fastT_gpu(tmp+i*N, vec_x+i*N, cuHandles, &fft_buffer, D_kx, D_ky, D_kz, Nx, Ny, Nz, -1);

 	vp_add_vp_add_vp<<<DimGrid, DimBlock>>>(N, Nd, 2*Nd, Pi_Prs, tmp+(N-Nd), vec_y);
 	vp_add_vp_add_vp<<<DimGrid, DimBlock>>>(N, Nd, 2*Nd, Pi_Prs+Nd, tmp+(N-Nd), vec_y+Nd);

    cudaFree(tmp);
    return 0;

}

static __global__ void vp_add_vp_add_vp(int N, int Nd, int Nd_2, cmpxGPU* L, cmpxGPU* vec, cmpxGPU* vec_out)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < Nd)
    {
        //vec_out[idx] = L_1[idx]*vec_1[idx] + L_3[idx]*vec_3[idx] + L_2[idx]*vec_2[idx]
		
        vec_out[idx].x = L[idx].x*vec[idx].x + L[idx+Nd_2].x*vec[idx+N].x + L[idx+2*Nd_2].x*vec[idx+2*N].x\
                         - L[idx].y*vec[idx].y - L[idx+Nd_2].y*vec[idx+N].y - L[idx+2*Nd_2].y*vec[idx+2*N].y;

        vec_out[idx].y = L[idx].x*vec[idx].y + L[idx+Nd_2].x*vec[idx+N].y + L[idx+2*Nd_2].x*vec[idx+2*N].y\
                         + L[idx].y*vec[idx].x + L[idx+Nd_2].y*vec[idx+N].x + L[idx+2*Nd_2].y*vec[idx+2*N].x;

    }

}

