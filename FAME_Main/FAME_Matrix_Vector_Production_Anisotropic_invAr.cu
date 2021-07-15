#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include "CG_Aniso.cuh"

static __global__ void pointwise_div_Nd8(cmpxGPU* vec_y, realGPU* Lambda_q_sqrt, int size);
static __global__ void pointwise_div_Nd8(cmpxGPU* vec_y, cmpxGPU* vec_x, realGPU* Lambda_q_sqrt, int size);

int FAME_Matrix_Vector_Production_Anisotropic_invAr(
	cmpxGPU* vec_y,
	cmpxGPU* vec_x,
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

	cmpxGPU* tmp = cuHandles.Nd2_temp1;

	pointwise_div_Nd8<<<DimGrid, DimBlock>>>(tmp, vec_x, Lambdas_cuda.Lambda_q_sqrt, Nd2);

	int iter;
	// Time start 
	struct timespec start, end;
	clock_gettime (CLOCK_REALTIME, &start);

	// Solve linear system for QBQ*y = x
    if(flag_CompType == "Simple")
    {
		iter = CG_Aniso(vec_y, tmp, cuHandles, fft_buffer, mtx_B, Lambdas_cuda.dD_k, Lambdas_cuda.dD_ks, 
                        Lambdas_cuda.dPi_Qr, Lambdas_cuda.dPi_Qrs, Lambdas_cuda.dPi_Qr_110, Lambdas_cuda.dPi_Qrs_110, 
                        Lambdas_cuda.dPi_Qr_101, Lambdas_cuda.dPi_Qrs_101, Lambdas_cuda.dPi_Qr_011, Lambdas_cuda.dPi_Qrs_011, 
                        Nx, Ny, Nz, Nd, ls.maxit, ls.tol, Profile);
    }
	else if(flag_CompType == "General")
    {
        iter = CG_Aniso(vec_y, tmp, cuHandles, fft_buffer, mtx_B, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz,  
            			Lambdas_cuda.dPi_Qr, Lambdas_cuda.dPi_Qrs, Lambdas_cuda.dPi_Qr_110, Lambdas_cuda.dPi_Qrs_110, 
            			Lambdas_cuda.dPi_Qr_101, Lambdas_cuda.dPi_Qrs_101, Lambdas_cuda.dPi_Qr_011, Lambdas_cuda.dPi_Qrs_011, 
            			Nx, Ny, Nz, Nd, ls.maxit, ls.tol, Profile);
    }

	// Time end 
	clock_gettime (CLOCK_REALTIME, &end);
	Profile->ls_iter[Profile->idx] += iter;
	Profile->ls_time[Profile->idx] += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / BILLION;
	Profile->es_iter[Profile->idx]++;

	pointwise_div_Nd8<<<DimGrid, DimBlock>>>(vec_y, Lambdas_cuda.Lambda_q_sqrt, Nd2);

	return 0;
}

static __global__ void pointwise_div_Nd8(cmpxGPU* vec_y, realGPU* Lambda_q_sqrt, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
    {
		for( int i = 0; i < 4; i++)
        {
        	vec_y[idx + i * size].x = vec_y[idx + i * size].x / Lambda_q_sqrt[idx];
			vec_y[idx + i * size].y = vec_y[idx + i * size].y / Lambda_q_sqrt[idx];
		}
    }

}

static __global__ void pointwise_div_Nd8(cmpxGPU* vec_y, cmpxGPU* vec_x, realGPU* Lambda_q_sqrt, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
    {
		for( int i = 0; i < 4; i++)
        {
        	vec_y[idx + i * size].x = vec_x[idx + i * size].x / Lambda_q_sqrt[idx];
			vec_y[idx + i * size].y = vec_x[idx + i * size].y / Lambda_q_sqrt[idx];
		}
    }

}
