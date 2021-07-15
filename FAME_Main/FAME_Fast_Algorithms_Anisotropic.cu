#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include "Lanczos_Anisotropic.cuh"
#include "FAME_Matrix_Vector_Production_Qr.cuh"
#include "FAME_Matrix_Vector_Production_Anisotropic_N_Shuffle.cuh"
#include "printDeviceArray.cuh"

static __global__ void initialize_aniso(cmpxGPU* vec, realCPU real, realCPU imag, int size);
static __global__ void dot_product_Nd8(cmpxGPU* vec_y, realCPU* array, int size);

int Eigen_Restoration_Anisotropic(
	cmpxGPU* Output_eigvec_mat,
	cmpxGPU* Input_eigvec_mat,
	CULIB_HANDLES    cuHandles,
	FFT_BUFFER       fft_buffer,
	LAMBDAS_CUDA     Lambdas_cuda,
	MTX_B            mtx_B,
	int Nx, int Ny, int Nz, int Nd, int N, int Nwant,
	string flag_CompType, PROFILE* Profile);
	

int FAME_Fast_Algorithms_Anisotropic(
	realCPU*        Freq_array,
	cmpxCPU*        Ele_field_mtx,
	cmpxCPU*        Dis_field_mtx,
	CULIB_HANDLES   cuHandles,
	LANCZOS_BUFFER  lBuffer,
	FFT_BUFFER      fft_buffer,
	LAMBDAS_CUDA    Lambdas_cuda,
	MTX_B           mtx_B,
	ES 			    es,
	LS 			    ls,
	int Nx, int Ny, int Nz, int Nd, int N,
	string flag_CompType, PROFILE* Profile)
{
	int N12 = 12 * N;
	int Nd8 = Nd * 8;
	size_t memsize;
	// Creat temp vector
	memsize = Nd8 * (es.nstep + 1) * sizeof(cmpxGPU);
	checkCudaErrors(cudaMalloc((void**) &lBuffer.dU, memsize));

	cmpxGPU* ev;
	checkCudaErrors(cudaMalloc((void**)&ev, Nd8 * ( es.nwant+2 ) * sizeof(cmpxGPU)));

    memsize = Nd8 * sizeof(cmpxGPU);
    checkCudaErrors(cudaMalloc((void**)&cuHandles.Nd2_temp1, memsize));
    checkCudaErrors(cudaMalloc((void**)&cuHandles.Nd2_temp2, memsize));
    checkCudaErrors(cudaMalloc((void**)&cuHandles.Nd2_temp3, memsize));
	checkCudaErrors(cudaMalloc((void**)&cuHandles.Nd2_temp4, memsize));

    Lanczos_Anisotropic(Freq_array, ev, cuHandles, lBuffer, fft_buffer, Lambdas_cuda, mtx_B, es, ls,
					   	                                  Nx, Ny, Nz, Nd, flag_CompType, Profile);

    cudaFree(lBuffer.dU);

    cmpxGPU* ev_back;
    checkCudaErrors(cudaMalloc((void**)&ev_back, N12 * es.nwant * sizeof(cmpxGPU)));


	Eigen_Restoration_Anisotropic(ev_back, ev, cuHandles, fft_buffer, Lambdas_cuda, mtx_B, 
						           Nx, Ny, Nz, Nd, N, es.nwant, flag_CompType, Profile);

	if(Nd == N - 1)
	{
		for(int i = es.nwant - 1; i >= 8 ; i--)
		{
			FAME_cublas_swap(cuHandles.cublas_handle, N12, ev_back + i * N12, 1, ev_back + (i - 8) * N12, 1);
			Freq_array[i] = Freq_array[i - 8];
		}

		Freq_array[0] = 0.0;
		Freq_array[1] = 0.0;
		Freq_array[2] = 0.0;
		Freq_array[3] = 0.0;
		Freq_array[4] = 0.0;
		Freq_array[5] = 0.0;
		Freq_array[6] = 0.0;
		Freq_array[7] = 0.0;

		dim3 DimBlock(BLOCK_SIZE, 1, 1);
		dim3 DimGrid((N12 - 1) / BLOCK_SIZE + 1, 1, 1);

		realCPU temp = 1.0 / sqrt(N12);
		initialize_aniso<<<DimGrid, DimBlock>>>(ev_back, temp, 0.0, N12);
		initialize_aniso<<<DimGrid, DimBlock>>>(ev_back + N12, temp, 0.0, N12);
		initialize_aniso<<<DimGrid, DimBlock>>>(ev_back + 2 * N12, temp, 0.0, N12);
		initialize_aniso<<<DimGrid, DimBlock>>>(ev_back + 3 * N12, temp, 0.0, N12);
		initialize_aniso<<<DimGrid, DimBlock>>>(ev_back + 4 * N12, temp, 0.0, N12);
		initialize_aniso<<<DimGrid, DimBlock>>>(ev_back + 5 * N12, temp, 0.0, N12);
		initialize_aniso<<<DimGrid, DimBlock>>>(ev_back + 6 * N12, temp, 0.0, N12);
		initialize_aniso<<<DimGrid, DimBlock>>>(ev_back + 7 * N12, temp, 0.0, N12);
	}

	checkCudaErrors(cudaMemcpy(Dis_field_mtx, ev_back, N12 * es.nwant * sizeof(cmpxGPU), cudaMemcpyDeviceToHost));

	int ii = 0;
	if(Nd == N-1)
	{
		ii = 8;
	}
	for(int i = ii; i < es.nwant; i++)
	{
		FAME_Matrix_Vector_Production_Anisotropic_N_Shuffle(cuHandles, Nx, Ny, Nz, Nd, mtx_B.GInOut_index, mtx_B.GInOut_index_length, mtx_B.N, ev_back+i*N12);
	}

	checkCudaErrors(cudaMemcpy(Ele_field_mtx, ev_back, N12 * es.nwant * sizeof(cmpxGPU), cudaMemcpyDeviceToHost));
	cudaFree(ev); cudaFree(ev_back);cudaFree(lBuffer.dU);
	cudaFree(cuHandles.Nd2_temp1); cudaFree(cuHandles.Nd2_temp2); cudaFree(cuHandles.Nd2_temp3); cudaFree(cuHandles.Nd2_temp4);
	return 0;
}



int Eigen_Restoration_Anisotropic(
	cmpxGPU* Output_eigvec_mat,
	cmpxGPU* Input_eigvec_mat,
	CULIB_HANDLES    cuHandles,
	FFT_BUFFER       fft_buffer,
	LAMBDAS_CUDA     Lambdas_cuda,
	MTX_B            mtx_B,
	int Nx, int Ny, int Nz, int Nd, int N, int Nwant,
	string flag_CompType, PROFILE* Profile)
{
	int N12 = N * 12;
	int Nd8 = Nd * 8;
	int N3 = N * 3;
	int Nd2 = Nd * 2;
	realCPU norm;

	dim3 DimBlock(BLOCK_SIZE, 1, 1);
	dim3 DimGrid((Nd2-1)/BLOCK_SIZE + 1, 1, 1);
	
	for(int i = 0; i < Nwant; i++)
	{
		
		dot_product_Nd8<<<DimGrid, DimBlock>>>(Input_eigvec_mat+i*Nd8, Lambdas_cuda.Lambda_q_sqrt, Nd2);

		if (flag_CompType == "Simple")
		{
			FAME_Matrix_Vector_Production_Qr(Output_eigvec_mat+i*N12, Input_eigvec_mat+i*Nd8, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_k, Lambdas_cuda.dPi_Qr );
			FAME_Matrix_Vector_Production_Qr(Output_eigvec_mat+i*N12+N3, Input_eigvec_mat+i*Nd8+Nd2, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_k, Lambdas_cuda.dPi_Qr_110 );
			FAME_Matrix_Vector_Production_Qr(Output_eigvec_mat+i*N12+2*N3, Input_eigvec_mat+i*Nd8+2*Nd2, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_k, Lambdas_cuda.dPi_Qr_101 );
			FAME_Matrix_Vector_Production_Qr(Output_eigvec_mat+i*N12+3*N3, Input_eigvec_mat+i*Nd8+3*Nd2, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_k, Lambdas_cuda.dPi_Qr_011 );
		}
		else if (flag_CompType == "General")
		{
			FAME_Matrix_Vector_Production_Qr(Output_eigvec_mat+i*N12, Input_eigvec_mat+i*Nd8, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Qr );
			FAME_Matrix_Vector_Production_Qr(Output_eigvec_mat+i*N12+N3, Input_eigvec_mat+i*Nd8+Nd2, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Qr_110 );
			FAME_Matrix_Vector_Production_Qr(Output_eigvec_mat+i*N12+2*N3, Input_eigvec_mat+i*Nd8+2*Nd2, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Qr_101 );
			FAME_Matrix_Vector_Production_Qr(Output_eigvec_mat+i*N12+3*N3, Input_eigvec_mat+i*Nd8+3*Nd2, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Qr_011 );
		}

		// FAME_Matrix_Vector_Production_Anisotropic_N_Shuffle(cuHandles, Nx, Ny, Nz, Nd, mtx_B, Output_eigvec_mat+i*N12);

		FAME_cublas_nrm2(cuHandles.cublas_handle, N12, Output_eigvec_mat+N12*i, 1, &norm);
   		norm = 1.0 / norm;
		FAME_cublas_dscal(cuHandles.cublas_handle, N12, &norm, Output_eigvec_mat+N12*i, 1);

	}
	
	return 0;
}

static __global__ void dot_product_Nd8(cmpxGPU* vec_y, realCPU* array, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
    {
		for( int i = 0; i < 4; i++)
        {
        	vec_y[idx + i * size].x = vec_y[idx + i * size].x * array[idx];
			vec_y[idx + i * size].y = vec_y[idx + i * size].y * array[idx];
		}
    }

}

static __global__ void initialize_aniso(cmpxGPU* vec, realCPU real, realCPU imag, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size)
	{
		vec[idx].x = real;
		vec[idx].y = imag;
	}
}

