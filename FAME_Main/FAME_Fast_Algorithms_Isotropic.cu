#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include "Lanczos_Isotropic.cuh"
#include "FAME_Matrix_Vector_Production_Qr.cuh"

static __global__ void initialize(cuDoubleComplex* vec, double real, double imag, int size);
static __global__ void dot_product(cuDoubleComplex* vec_y, double* array, int size);


int Eigen_Restoration_Isotropic(
	cuDoubleComplex* Output_eigvec_mat,
	cuDoubleComplex* Input_eigvec_mat,
	CULIB_HANDLES    cuHandles,
	FFT_BUFFER       fft_buffer,
	LAMBDAS_CUDA     Lambdas_cuda,
	MTX_B            mtx_B,
	int Nx, int Ny, int Nz, int Nd, int N, int Nwant,
	string flag_CompType, PROFILE* Profile);

int FAME_Fast_Algorithms_Isotropic(
	double*        Freq_array,
	cmpx*          Ele_field_mtx,
	CULIB_HANDLES  cuHandles,
	LANCZOS_BUFFER lBuffer,
	FFT_BUFFER     fft_buffer,
	LAMBDAS_CUDA   Lambdas_cuda,
	MTX_B          mtx_B,
	ES 			   es,
	LS 			   ls,
	int Nx, int Ny, int Nz, int Nd, int N,
	string flag_CompType, PROFILE* Profile)
{
	int N3 = 3 * N;
	int Nd2 = Nd * 2;
	size_t memsize;
	// Creat temp vector
	memsize = Nd2 * (es.nstep + 1) * sizeof(cuDoubleComplex);
	checkCudaErrors(cudaMalloc((void**) &lBuffer.dU, memsize));

	cuDoubleComplex* ev;
	checkCudaErrors(cudaMalloc((void**)&ev, Nd * 2 * ( es.nwant+2 ) * sizeof(cuDoubleComplex)));

    memsize = Nd2 * sizeof(cuDoubleComplex);
    checkCudaErrors(cudaMalloc((void**)&cuHandles.Nd2_temp1, memsize));
    checkCudaErrors(cudaMalloc((void**)&cuHandles.Nd2_temp2, memsize));
    checkCudaErrors(cudaMalloc((void**)&cuHandles.Nd2_temp3, memsize));
	checkCudaErrors(cudaMalloc((void**)&cuHandles.Nd2_temp4, memsize));

    Lanczos_Isotropic(Freq_array, ev, cuHandles, lBuffer, fft_buffer, Lambdas_cuda, mtx_B, es, ls,
					   	                                  Nx, Ny, Nz, Nd, flag_CompType, Profile);

    cudaFree(lBuffer.dU);

    cuDoubleComplex* ev_back;
    checkCudaErrors(cudaMalloc((void**)&ev_back, N3 * es.nwant * sizeof(cuDoubleComplex)));


	Eigen_Restoration_Isotropic(ev_back, ev, cuHandles, fft_buffer, Lambdas_cuda, mtx_B, 
						           Nx, Ny, Nz, Nd, N, es.nwant, flag_CompType, Profile);

	if(Nd == N-1)
	{
		for(int i = es.nwant - 1; i >= 2 ; i--)
		{
			cublasZswap_v2(cuHandles.cublas_handle, N3, ev_back + i * N3, 1, ev_back + (i - 2) * N3, 1);
			Freq_array[i] = Freq_array[i - 2];
		}

		Freq_array[0] = 0.0;
		Freq_array[1] = 0.0;

		double temp = 1.0 / sqrt(N3);

		dim3 DimBlock(BLOCK_SIZE, 1, 1);
		dim3 DimGrid((N3-1)/BLOCK_SIZE + 1, 1, 1);

		initialize<<<DimGrid, DimBlock>>>(ev_back,      temp, 0.0, N3);
		initialize<<<DimGrid, DimBlock>>>(ev_back + N3, temp, 0.0, N3);
	}

	checkCudaErrors(cudaMemcpy(Ele_field_mtx, ev_back, N3 * es.nwant * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
	
	cudaFree(ev); cudaFree(ev_back);
	cudaFree(cuHandles.Nd2_temp1); cudaFree(cuHandles.Nd2_temp2); cudaFree(cuHandles.Nd2_temp3); cudaFree(cuHandles.Nd2_temp4);
	return 0;
}



int Eigen_Restoration_Isotropic(
	cuDoubleComplex* Output_eigvec_mat,
	cuDoubleComplex* Input_eigvec_mat,
	CULIB_HANDLES    cuHandles,
	FFT_BUFFER       fft_buffer,
	LAMBDAS_CUDA     Lambdas_cuda,
	MTX_B            mtx_B,
	int Nx, int Ny, int Nz, int Nd, int N, int Nwant,
	string flag_CompType, PROFILE* Profile)
{
	int N3 = N * 3;
	int Nd2 = Nd * 2;
	double norm;

	dim3 DimBlock(BLOCK_SIZE, 1, 1);
	dim3 DimGrid((N3-1)/BLOCK_SIZE + 1, 1, 1);
	
	for(int i = 0; i < Nwant; i++)
	{
		
		dot_product<<<DimGrid, DimBlock>>>(Input_eigvec_mat+i*Nd2, Lambdas_cuda.Lambda_q_sqrt, Nd2);

		if (flag_CompType == "Simple")
			FAME_Matrix_Vector_Production_Qr(Output_eigvec_mat+i*N3, Input_eigvec_mat+i*Nd2, cuHandles, fft_buffer, Lambdas_cuda.dD_k, Lambdas_cuda.dPi_Qr, Nx, Ny, Nz, Nd, Profile );
		
		else if (flag_CompType == "General")
			FAME_Matrix_Vector_Production_Qr(Output_eigvec_mat+i*N3, Input_eigvec_mat+i*Nd2, cuHandles, fft_buffer, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Qr, Nx, Ny, Nz, Nd, Profile );

		dot_product<<<DimGrid, DimBlock>>>(Output_eigvec_mat+i*N3, mtx_B.invB_eps, N3);

		cublasDznrm2_v2(cuHandles.cublas_handle, N3, Output_eigvec_mat+N3*i, 1, &norm);

		cublasZdscal_v2(cuHandles.cublas_handle, N3, &norm, Output_eigvec_mat+N3*i, 1);
	}
	
	return 0;
}

static __global__ void dot_product(cuDoubleComplex* vec_y, double* array, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
    {
        vec_y[idx].x = vec_y[idx].x * array[idx];
		vec_y[idx].y = vec_y[idx].y * array[idx];
    }

}

static __global__ void initialize(cuDoubleComplex* vec, double real, double imag, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size)
	{
		vec[idx].x = real;
		vec[idx].y = imag;
	}
}
