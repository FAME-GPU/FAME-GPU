#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include "CG.cuh"

static __global__ void pointwise_div(cuDoubleComplex* vec_y, double* Lambda_q_sqrt, int size);
static __global__ void pointwise_div(cuDoubleComplex* vec_y, cuDoubleComplex* vec_x, double* Lambda_q_sqrt, int size);

int FAME_Matrix_Vector_Production_Isotropic_invAr(
	cuDoubleComplex* vec_y,
	cuDoubleComplex* vec_x,
	CULIB_HANDLES    cuHandles,
	FFT_BUFFER       fft_buffer,
	LAMBDAS_CUDA     Lambdas_cuda,
	MTX_B            mtx_B,
	LS               ls,
	int Nx, int Ny, int Nz, int Nd,
	string flag_CompType, PROFILE* Profile)
{
	int Nd2 = Nd * 2;

	dim3 DimBlock(BLOCK_SIZE, 1, 1);
	dim3 DimGrid((Nd2-1)/BLOCK_SIZE + 1, 1, 1 );

	cuDoubleComplex* tmp = cuHandles.Nd2_temp1;

	pointwise_div<<<DimGrid, DimBlock>>>(tmp, vec_x, Lambdas_cuda.Lambda_q_sqrt, Nd2);

	int iter;
	// Time start 
	struct timespec start, end;
	clock_gettime (CLOCK_REALTIME, &start);

	// Solve linear system for QBQ*y = x
    if(flag_CompType == "Simple")
		iter = CG(vec_y, tmp, cuHandles, fft_buffer, mtx_B, Lambdas_cuda.dD_k, Lambdas_cuda.dD_ks, Lambdas_cuda.dPi_Qr, Lambdas_cuda.dPi_Qrs, Nx, Ny, Nz, Nd, ls.maxit, ls.tol, Profile);
	else if(flag_CompType == "General")
		iter = CG(vec_y, tmp, cuHandles, fft_buffer, mtx_B, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Qr, Lambdas_cuda.dPi_Qrs, Nx, Ny, Nz, Nd, ls.maxit, ls.tol, Profile);

	// Time end 
	clock_gettime (CLOCK_REALTIME, &end);
	Profile->ls_iter[Profile->idx] += iter;
	Profile->ls_time[Profile->idx] += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / BILLION;
	Profile->es_iter[Profile->idx]++;

	pointwise_div<<<DimGrid, DimBlock>>>(vec_y, Lambdas_cuda.Lambda_q_sqrt, Nd2);

	return 0;
}

static __global__ void pointwise_div(cuDoubleComplex* vec_y, double* Lambda_q_sqrt, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
    {
        vec_y[idx].x = vec_y[idx].x / Lambda_q_sqrt[idx];
		vec_y[idx].y = vec_y[idx].y / Lambda_q_sqrt[idx];
    }

}

static __global__ void pointwise_div(cuDoubleComplex* vec_y, cuDoubleComplex* vec_x, double* Lambda_q_sqrt, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
    {
        vec_y[idx].x = vec_x[idx].x / Lambda_q_sqrt[idx];
		vec_y[idx].y = vec_x[idx].y / Lambda_q_sqrt[idx];
    }

}