#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include "FAME_FFT_CUDA.cuh"
#include "printDeviceArray.cuh"

static __global__ void vp_add_vp(int size, cmpxGPU* L_1, cmpxGPU* L_2, cmpxGPU* vec_1, cmpxGPU* vec_2,cmpxGPU* vec_out);
////////////=========================== Create Qr function for Biiso (cuda)===========================//////////////////

int FAME_Matrix_Vector_Production_Qr(
	cmpxGPU* vec_y,
	cmpxGPU* vec_x,
	CULIB_HANDLES    cuHandles, 
	FFT_BUFFER       fft_buffer, 
	int Nx, int Ny, int Nz, int Nd,
	cmpxGPU* D_k, 
	cmpxGPU* Pi_Qr)
{
    int N  = Nx * Ny * Nz;
    int N2 = N * 2;
    int N3 = N * 3;

	dim3 DimBlock(BLOCK_SIZE, 1, 1);
    dim3 DimGrid((Nd - 1)/BLOCK_SIZE +1, 1, 1);	
	
	// cuHandles.N3_temp2 is used in QBQ
	cmpxGPU* temp = cuHandles.N3_temp2;	
	

	checkCudaErrors(cudaMemset(temp, 0, N3 * sizeof(cmpxGPU)));

    //temp = Pi_Qr * vec_x
    vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Qr,      Pi_Qr+3*Nd, vec_x, vec_x+Nd, temp+N-Nd);
    vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Qr+Nd,   Pi_Qr+4*Nd, vec_x, vec_x+Nd, temp+N2-Nd);
    vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Qr+2*Nd, Pi_Qr+5*Nd, vec_x, vec_x+Nd, temp+N3-Nd);

	IFFT_CUDA(vec_y, temp, D_k, fft_buffer, cuHandles, Nx, Ny, Nz);

	return 0;
}

int FAME_Matrix_Vector_Production_Qr(
	cmpxGPU* vec_y,
	cmpxGPU* vec_x,
	CULIB_HANDLES cuHandles, 
	FFT_BUFFER fft_buffer, 
	int Nx, int Ny, int Nz, int Nd,
	cmpxGPU* D_kx,
	cmpxGPU* D_ky,
	cmpxGPU* D_kz, 
	cmpxGPU* Pi_Qr)
{
    int N  = Nx * Ny * Nz;
    int N2 = N * 2;
    int N3 = N * 3;
    // cuHandles.N3_temp2 is used in QBQ
    cmpxGPU* temp = cuHandles.N3_temp2;
	//cmpxGPU* temp;
	//checkCudaErrors(cudaMalloc((void**)&temp, N3*sizeof(cmpxGPU)));

    dim3 DimBlock(BLOCK_SIZE, 1, 1);
    dim3 DimGrid((Nd - 1)/BLOCK_SIZE +1, 1, 1);

	// Initial
	checkCudaErrors(cudaMemset(temp, 0, N3 * sizeof(cmpxGPU)));
	
	//Pi_Qr * vec_x
	vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Qr,      Pi_Qr+3*Nd, vec_x, vec_x+Nd, temp+N-Nd);
    vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Qr+Nd,   Pi_Qr+4*Nd, vec_x, vec_x+Nd, temp+N2-Nd);
    vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Qr+2*Nd, Pi_Qr+5*Nd, vec_x, vec_x+Nd, temp+N3-Nd);


	spMV_fastT_gpu(vec_y,    temp,    cuHandles, &fft_buffer, D_kx, D_ky, D_kz, Nx, Ny, Nz, 1);
	spMV_fastT_gpu(vec_y+N,  temp+N,  cuHandles, &fft_buffer, D_kx, D_ky, D_kz, Nx, Ny, Nz, 1);
	spMV_fastT_gpu(vec_y+N2, temp+N2, cuHandles, &fft_buffer, D_kx, D_ky, D_kz, Nx, Ny, Nz, 1);
	
	return 0;
}

// temp = Pi_Qr_1 dot vec_1 + Pi_Qr_2 dot vec_2
static __global__ void vp_add_vp(int size, cmpxGPU* L_1, cmpxGPU* L_2, cmpxGPU* vec_1, cmpxGPU* vec_2,cmpxGPU* vec_out)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < size)
    {
        vec_out[idx].x = L_1[idx].x*vec_1[idx].x + L_2[idx].x*vec_2[idx].x - L_1[idx].y*vec_1[idx].y - L_2[idx].y*vec_2[idx].y;
        vec_out[idx].y = L_1[idx].x*vec_1[idx].y + L_2[idx].y*vec_2[idx].x + L_1[idx].y*vec_1[idx].x + L_2[idx].x*vec_2[idx].y;

    }

}

