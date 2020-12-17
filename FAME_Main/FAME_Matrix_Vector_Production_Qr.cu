#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include "FAME_FFT_CUDA.cuh"
#include "printDeviceArray.cuh"

static __global__ void vp_add_vp(int size, cuDoubleComplex* L_1, cuDoubleComplex* L_2, cuDoubleComplex* vec_1, cuDoubleComplex* vec_2,cuDoubleComplex* vec_out);
////////////=========================== Create Qr function for Biiso (cuda)===========================//////////////////

int FAME_Matrix_Vector_Production_Qr(
	cuDoubleComplex* vec_y,
	cuDoubleComplex* vec_x,
	CULIB_HANDLES    cuHandles, 
	FFT_BUFFER       fft_buffer, 
	cuDoubleComplex* D_k, 
	cuDoubleComplex* Pi_Qr,
	int Nx, int Ny, int Nz, int Nd, PROFILE* Profile)
{
    int N  = Nx * Ny * Nz;
    int N2 = N * 2;
    int N3 = N * 3;

	dim3 DimBlock(BLOCK_SIZE, 1, 1);
    dim3 DimGrid((Nd - 1)/BLOCK_SIZE +1, 1, 1);	
	
	// cuHandles.N3_temp2 is used in QBQ
	cuDoubleComplex* temp = cuHandles.N3_temp2;	
	

	checkCudaErrors(cudaMemset(temp, 0, N3 * sizeof(cuDoubleComplex)));

    //temp = Pi_Qr * vec_x
    vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Qr,      Pi_Qr+3*Nd, vec_x, vec_x+Nd, temp+N-Nd);
    vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Qr+Nd,   Pi_Qr+4*Nd, vec_x, vec_x+Nd, temp+N2-Nd);
    vp_add_vp<<<DimGrid, DimBlock>>>(Nd, Pi_Qr+2*Nd, Pi_Qr+5*Nd, vec_x, vec_x+Nd, temp+N3-Nd);

	/*CUDA version (follow matlab) correct*/
//	struct timespec start, end;
//	clock_gettime (CLOCK_REALTIME, &start);
	IFFT_CUDA(vec_y, temp, D_k, fft_buffer, cuHandles, Nx, Ny, Nz);
	// Time end 
//	clock_gettime (CLOCK_REALTIME, &end);	
//	Profile->ifft_time[Profile->idx] += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / BILLION;

	return 0;
}

int FAME_Matrix_Vector_Production_Qr(
	cuDoubleComplex* vec_y,
	cuDoubleComplex* vec_x,
	CULIB_HANDLES    cuHandles, 
	FFT_BUFFER       fft_buffer, 
	cuDoubleComplex* D_kx, 
	cuDoubleComplex* D_ky, 
	cuDoubleComplex* D_kz, 
	cuDoubleComplex* Pi_Qr,
	int Nx, int Ny, int Nz, int Nd,PROFILE* Profile)
{
    int N  = Nx * Ny * Nz;
    int N2 = N * 2;
    int N3 = N * 3;

    // cuHandles.N3_temp2 is used in QBQ
    cuDoubleComplex* temp = cuHandles.N3_temp2;
	//cuDoubleComplex* temp;
	//checkCudaErrors(cudaMalloc((void**)&temp, N3*sizeof(cuDoubleComplex)));

    dim3 DimBlock(BLOCK_SIZE, 1, 1);
    dim3 DimGrid((Nd - 1)/BLOCK_SIZE +1, 1, 1);

	// Initial
	checkCudaErrors(cudaMemset(temp, 0, N3 * sizeof(cuDoubleComplex)));
	

	
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
static __global__ void vp_add_vp(int size, cuDoubleComplex* L_1, cuDoubleComplex* L_2, cuDoubleComplex* vec_1, cuDoubleComplex* vec_2,cuDoubleComplex* vec_out)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < size)
    {
        vec_out[idx].x = L_1[idx].x*vec_1[idx].x + L_2[idx].x*vec_2[idx].x - L_1[idx].y*vec_1[idx].y - L_2[idx].y*vec_2[idx].y;
        vec_out[idx].y = L_1[idx].x*vec_1[idx].y + L_2[idx].y*vec_2[idx].x + L_1[idx].y*vec_1[idx].x + L_2[idx].x*vec_2[idx].y;

    }

}

