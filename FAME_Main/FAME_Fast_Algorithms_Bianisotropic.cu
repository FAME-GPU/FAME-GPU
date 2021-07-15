#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include <complex.h>

#include "FAME_Matrix_Vector_Production_Pr.cuh"
#include "FAME_Matrix_Vector_Production_Qr.cuh"
#include "FAME_Matrix_Vector_Production_Bianisotropic_G_Shuffle.cuh"
#include "FAME_Matrix_Vector_Production_Bianisotropic_dualG_Shuffle.cuh"
#include "Lanczos_Bianisotropic.cuh"
#include "printDeviceArray.cuh"


int Eigen_Restoration_Bianisotropic(  CULIB_HANDLES cuHandles,
                                    LAMBDAS_CUDA Lambdas_cuda,
                                    FFT_BUFFER    fft_buffer,
                                    cmpxGPU* Input_eigvec_mat,
                                    cmpxGPU* Output_eigvec_mat,
                                    MTX_B  mtx_B,
                                    int N_eig_wanted,
                                    int Nx,
                                    int Ny,
                                    int Nz,
                                    int Nd,
                                    string flag_CompType, PROFILE* Profile);
static __global__ void initialize_bianiso(cmpxGPU* vec, realCPU real, realCPU imag, int size);
 

int FAME_Fast_Algorithms_Bianisotropic
	(realCPU* Freq_array,
	 cmpxCPU* Ele_field_mtx,
	 cmpxCPU* Dis_field_mtx,
	 CULIB_HANDLES cuHandles,
	 LAMBDAS_CUDA  Lambdas_cuda, 
	 LANCZOS_BUFFER lBuffer,
	 FFT_BUFFER    fft_buffer,
	 MTX_B 	mtx_B,
	 MATERIAL material,	 
	 int Nx, 
	 int Ny, 
	 int Nz,
	 int Nd,
	 ES  es,
	 LS  ls, 
	 string flag_CompType,
	 PROFILE* Profile)
{
	int N = Nx*Ny*Nz;
	int Nd32 = Nd * 32;
	int N48 = 48 * N;
	size_t memsize;
	int eigen_wanted = es.nwant;
	cublasStatus_t cublasStatus;

	memsize = Nd32 * sizeof(cmpxGPU);
    checkCudaErrors(cudaMalloc((void**)&cuHandles.Nd2_temp1, memsize));
    checkCudaErrors(cudaMalloc((void**)&cuHandles.Nd2_temp2, memsize));
    checkCudaErrors(cudaMalloc((void**)&cuHandles.Nd2_temp3, memsize));
	checkCudaErrors(cudaMalloc((void**)&cuHandles.Nd2_temp4, memsize));

	memsize = Nd32 * (es.nstep + 1) * sizeof(cmpxGPU);
	checkCudaErrors(cudaMalloc((void**)&lBuffer.dU, memsize));
	
	cmpxGPU* DEV_Back;
	checkCudaErrors(cudaMalloc((void**)&DEV_Back, sizeof(cmpxGPU)*N48*eigen_wanted));
	cmpxGPU* DEV;
	checkCudaErrors(cudaMalloc((void**)&DEV, sizeof(cmpxGPU)*Nd32*(eigen_wanted+2)*2));

	Lanczos_Bianisotropic( cuHandles,
		fft_buffer, lBuffer,
		mtx_B, Nx, Ny, Nz, Nd, es, ls,
		Lambdas_cuda.Lambda_q_sqrt,
		Lambdas_cuda.dPi_Qr,
		Lambdas_cuda.dPi_Pr,
		Lambdas_cuda.dPi_Qrs,
		Lambdas_cuda.dPi_Prs,
		Lambdas_cuda.dPi_Qr_110,
		Lambdas_cuda.dPi_Pr_110,
		Lambdas_cuda.dPi_Qrs_110,
		Lambdas_cuda.dPi_Prs_110,
		Lambdas_cuda.dPi_Qr_101,
		Lambdas_cuda.dPi_Pr_101,
		Lambdas_cuda.dPi_Qrs_101,
		Lambdas_cuda.dPi_Prs_101,
		Lambdas_cuda.dPi_Qr_011,
		Lambdas_cuda.dPi_Pr_011,
		Lambdas_cuda.dPi_Qrs_011,
		Lambdas_cuda.dPi_Prs_011,
		Lambdas_cuda.dD_k,
		Lambdas_cuda.dD_ks,
		Lambdas_cuda.dD_kx,
		Lambdas_cuda.dD_ky,
		Lambdas_cuda.dD_kz,
		Freq_array, DEV,
		flag_CompType,
		Profile );
	

	cudaFree(lBuffer.dU);

	Eigen_Restoration_Bianisotropic(  cuHandles, Lambdas_cuda,
		fft_buffer,
		DEV, DEV_Back, 
		mtx_B, eigen_wanted,
		Nx, Ny, Nz, Nd,
		flag_CompType, Profile);

	if(Nd == N - 1)
	{
		for(int i = es.nwant - 1; i >= 16; i--)
		{
			cublasStatus = FAME_cublas_swap(cuHandles.cublas_handle, N48, DEV_Back + i * N48, 1, DEV_Back + (i - 16) * N48, 1);
			assert( cublasStatus == CUBLAS_STATUS_SUCCESS );
			Freq_array[i] = Freq_array[i - 16];
		}

		Freq_array[0] = 0.0;
		Freq_array[1] = 0.0;
		Freq_array[2] = 0.0;
		Freq_array[3] = 0.0;
		Freq_array[4] = 0.0;
		Freq_array[5] = 0.0;
		Freq_array[6] = 0.0;
		Freq_array[7] = 0.0;
		Freq_array[8] = 0.0;
		Freq_array[9] = 0.0;
		Freq_array[10] = 0.0;
		Freq_array[11] = 0.0;
		Freq_array[12] = 0.0;
		Freq_array[13] = 0.0;
		Freq_array[14] = 0.0;
		Freq_array[15] = 0.0;

		dim3 DimBlock(BLOCK_SIZE, 1, 1);
		dim3 DimGrid((N48-1)/BLOCK_SIZE + 1, 1, 1);

		realCPU temp = 1.0 / sqrt(N48);
		initialize_bianiso<<<DimGrid, DimBlock>>>(DEV_Back, temp, 0.0, N48);
		initialize_bianiso<<<DimGrid, DimBlock>>>(DEV_Back + N48, temp, 0.0, N48);
		initialize_bianiso<<<DimGrid, DimBlock>>>(DEV_Back + 2 * N48, temp, 0.0, N48);
		initialize_bianiso<<<DimGrid, DimBlock>>>(DEV_Back + 3 * N48, temp, 0.0, N48);
		initialize_bianiso<<<DimGrid, DimBlock>>>(DEV_Back + 4 * N48, temp, 0.0, N48);
		initialize_bianiso<<<DimGrid, DimBlock>>>(DEV_Back + 5 * N48, temp, 0.0, N48);
		initialize_bianiso<<<DimGrid, DimBlock>>>(DEV_Back + 6 * N48, temp, 0.0, N48);
		initialize_bianiso<<<DimGrid, DimBlock>>>(DEV_Back + 7 * N48, temp, 0.0, N48);
		initialize_bianiso<<<DimGrid, DimBlock>>>(DEV_Back + 8 * N48, temp, 0.0, N48);
		initialize_bianiso<<<DimGrid, DimBlock>>>(DEV_Back + 9 * N48, temp, 0.0, N48);
		initialize_bianiso<<<DimGrid, DimBlock>>>(DEV_Back + 10 * N48, temp, 0.0, N48);
		initialize_bianiso<<<DimGrid, DimBlock>>>(DEV_Back + 11 * N48, temp, 0.0, N48);
		initialize_bianiso<<<DimGrid, DimBlock>>>(DEV_Back + 12 * N48, temp, 0.0, N48);
		initialize_bianiso<<<DimGrid, DimBlock>>>(DEV_Back + 13 * N48, temp, 0.0, N48);
		initialize_bianiso<<<DimGrid, DimBlock>>>(DEV_Back + 14 * N48, temp, 0.0, N48);
		initialize_bianiso<<<DimGrid, DimBlock>>>(DEV_Back + 15 * N48, temp, 0.0, N48);
	}

	checkCudaErrors(cudaMemcpy(Dis_field_mtx, DEV_Back, N48 * eigen_wanted * sizeof(cmpxGPU), cudaMemcpyDeviceToHost));

	int ii = 0;
	if(Nd == N-1)
	{
		ii = 16;
	}
	for(int i = ii; i < eigen_wanted; i++)
	{
		FAME_Matrix_Vector_Production_Bianisotropic_G_Shuffle(cuHandles, Nx, Ny, Nz, Nd, mtx_B.GInOut_index, mtx_B.GInOut_index_length, mtx_B.G, DEV_Back+i*N48);
		FAME_Matrix_Vector_Production_Bianisotropic_dualG_Shuffle(cuHandles, Nx, Ny, Nz, Nd, mtx_B.GInOut_index, mtx_B.GInOut_index_length, mtx_B.G, DEV_Back+24*N+i*N48);
	}
	
	checkCudaErrors(cudaMemcpy(Ele_field_mtx, DEV_Back, N48 * eigen_wanted * sizeof(cmpxGPU), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(DEV)); 
	checkCudaErrors(cudaFree(DEV_Back));
	cudaFree(cuHandles.Nd2_temp1); cudaFree(cuHandles.Nd2_temp2); cudaFree(cuHandles.Nd2_temp3); cudaFree(cuHandles.Nd2_temp4);

	return 0;
}

int Eigen_Restoration_Bianisotropic(CULIB_HANDLES cuHandles,
									LAMBDAS_CUDA Lambdas_cuda,
									FFT_BUFFER    fft_buffer,
									cmpxGPU* Input_eigvec_mat, 
									cmpxGPU* Output_eigvec_mat,
									MTX_B  mtx_B,	
									int N_eig_wanted, 
									int Nx,
									int Ny,
									int Nz,
									int Nd,
									string flag_CompType,PROFILE* Profile)
{
	int N = Nx*Ny*Nz;
	realGPU norm_vec = 0.0;

	// Start to restore the eigenvectors

	int size = 2 * Nd;
	int size_temp = 3 * N;
	// Start to restore the eigenvectors
	for(int i = 0; i < N_eig_wanted; i++)
	{
		if (flag_CompType == "Simple")
		{
			FAME_Matrix_Vector_Production_Qr(Output_eigvec_mat+0*size_temp+i*48*N, Input_eigvec_mat+0*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_k, Lambdas_cuda.dPi_Qr);	
			FAME_Matrix_Vector_Production_Qr(Output_eigvec_mat+1*size_temp+i*48*N, Input_eigvec_mat+1*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_k, Lambdas_cuda.dPi_Qr_110);	
			FAME_Matrix_Vector_Production_Qr(Output_eigvec_mat+2*size_temp+i*48*N, Input_eigvec_mat+2*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_k, Lambdas_cuda.dPi_Qr_101);	
			FAME_Matrix_Vector_Production_Qr(Output_eigvec_mat+3*size_temp+i*48*N, Input_eigvec_mat+3*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_k, Lambdas_cuda.dPi_Qr_011);	

			FAME_Matrix_Vector_Production_Qr(Output_eigvec_mat+4*size_temp+i*48*N, Input_eigvec_mat+4*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_k, Lambdas_cuda.dPi_Qr);	
			FAME_Matrix_Vector_Production_Qr(Output_eigvec_mat+5*size_temp+i*48*N, Input_eigvec_mat+5*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_k, Lambdas_cuda.dPi_Qr_110);	
			FAME_Matrix_Vector_Production_Qr(Output_eigvec_mat+6*size_temp+i*48*N, Input_eigvec_mat+6*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_k, Lambdas_cuda.dPi_Qr_101);	
			FAME_Matrix_Vector_Production_Qr(Output_eigvec_mat+7*size_temp+i*48*N, Input_eigvec_mat+7*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_k, Lambdas_cuda.dPi_Qr_011);	

			// FAME_Matrix_Vector_Production_Bianisotropic_G_Shuffle(cuHandles, Nx, Ny, Nz, Nd, mtx_B.material_num, mtx_B.GInOut, mtx_B.GInOut_index, mtx_B.GInOut_index_length, mtx_B.G, Output_eigvec_mat+i*48*N);
		
			FAME_Matrix_Vector_Production_Pr(Output_eigvec_mat+8*size_temp+i*48*N, Input_eigvec_mat+8*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_k, Lambdas_cuda.dPi_Pr);	
			FAME_Matrix_Vector_Production_Pr(Output_eigvec_mat+9*size_temp+i*48*N, Input_eigvec_mat+9*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_k, Lambdas_cuda.dPi_Pr_110);	
			FAME_Matrix_Vector_Production_Pr(Output_eigvec_mat+10*size_temp+i*48*N, Input_eigvec_mat+10*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_k, Lambdas_cuda.dPi_Pr_101);	
			FAME_Matrix_Vector_Production_Pr(Output_eigvec_mat+11*size_temp+i*48*N, Input_eigvec_mat+11*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_k, Lambdas_cuda.dPi_Pr_011);	

			FAME_Matrix_Vector_Production_Pr(Output_eigvec_mat+12*size_temp+i*48*N, Input_eigvec_mat+12*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_k, Lambdas_cuda.dPi_Pr);	
			FAME_Matrix_Vector_Production_Pr(Output_eigvec_mat+13*size_temp+i*48*N, Input_eigvec_mat+13*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_k, Lambdas_cuda.dPi_Pr_110);	
			FAME_Matrix_Vector_Production_Pr(Output_eigvec_mat+14*size_temp+i*48*N, Input_eigvec_mat+14*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_k, Lambdas_cuda.dPi_Pr_101);	
			FAME_Matrix_Vector_Production_Pr(Output_eigvec_mat+15*size_temp+i*48*N, Input_eigvec_mat+15*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_k, Lambdas_cuda.dPi_Pr_011);	

			// FAME_Matrix_Vector_Production_Bianisotropic_dualG_Shuffle(cuHandles, Nx, Ny, Nz, Nd, mtx_B.material_num, mtx_B.GInOut, mtx_B.GInOut_index, mtx_B.GInOut_index_length, mtx_B.G, Output_eigvec_mat+8*size_temp+i*48*N);
		
		}
		else if (flag_CompType == "General")
		{
			FAME_Matrix_Vector_Production_Qr(Output_eigvec_mat+0*size_temp+i*48*N, Input_eigvec_mat+0*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Qr);	
			FAME_Matrix_Vector_Production_Qr(Output_eigvec_mat+1*size_temp+i*48*N, Input_eigvec_mat+1*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Qr_110);	
			FAME_Matrix_Vector_Production_Qr(Output_eigvec_mat+2*size_temp+i*48*N, Input_eigvec_mat+2*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Qr_101);	
			FAME_Matrix_Vector_Production_Qr(Output_eigvec_mat+3*size_temp+i*48*N, Input_eigvec_mat+3*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Qr_011);	

			FAME_Matrix_Vector_Production_Qr(Output_eigvec_mat+4*size_temp+i*48*N, Input_eigvec_mat+4*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Qr);	
			FAME_Matrix_Vector_Production_Qr(Output_eigvec_mat+5*size_temp+i*48*N, Input_eigvec_mat+5*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Qr_110);	
			FAME_Matrix_Vector_Production_Qr(Output_eigvec_mat+6*size_temp+i*48*N, Input_eigvec_mat+6*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Qr_101);	
			FAME_Matrix_Vector_Production_Qr(Output_eigvec_mat+7*size_temp+i*48*N, Input_eigvec_mat+7*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Qr_011);	

			// FAME_Matrix_Vector_Production_Bianisotropic_G_Shuffle(cuHandles, Nx, Ny, Nz, Nd, mtx_B.material_num, mtx_B.GInOut, mtx_B.GInOut_index, mtx_B.GInOut_index_length, mtx_B.G, Output_eigvec_mat+i*48*N);
		
			FAME_Matrix_Vector_Production_Pr(Output_eigvec_mat+8*size_temp+i*48*N, Input_eigvec_mat+8*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Pr);	
			FAME_Matrix_Vector_Production_Pr(Output_eigvec_mat+9*size_temp+i*48*N, Input_eigvec_mat+9*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Pr_110);	
			FAME_Matrix_Vector_Production_Pr(Output_eigvec_mat+10*size_temp+i*48*N, Input_eigvec_mat+10*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Pr_101);	
			FAME_Matrix_Vector_Production_Pr(Output_eigvec_mat+11*size_temp+i*48*N, Input_eigvec_mat+11*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Pr_011);	

			FAME_Matrix_Vector_Production_Pr(Output_eigvec_mat+12*size_temp+i*48*N, Input_eigvec_mat+12*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Pr);	
			FAME_Matrix_Vector_Production_Pr(Output_eigvec_mat+13*size_temp+i*48*N, Input_eigvec_mat+13*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Pr_110);	
			FAME_Matrix_Vector_Production_Pr(Output_eigvec_mat+14*size_temp+i*48*N, Input_eigvec_mat+14*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Pr_101);	
			FAME_Matrix_Vector_Production_Pr(Output_eigvec_mat+15*size_temp+i*48*N, Input_eigvec_mat+15*size+i*32*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Pr_011);	

			// FAME_Matrix_Vector_Production_Bianisotropic_dualG_Shuffle(cuHandles, Nx, Ny, Nz, Nd, mtx_B.material_num, mtx_B.GInOut, mtx_B.GInOut_index, mtx_B.GInOut_index_length, mtx_B.G, Output_eigvec_mat+8*size_temp+i*48*N);
		
		}

		// Normalize the eigenvector
		FAME_cublas_nrm2(cuHandles.cublas_handle, 48*N, Output_eigvec_mat+48*N*i, 1, &norm_vec);
		norm_vec = 1.0 / norm_vec;
		FAME_cublas_dscal(cuHandles.cublas_handle, 48*N, &norm_vec, Output_eigvec_mat+48*N*i, 1);
	}

	return 0;
}

static __global__ void initialize_bianiso(cmpxGPU* vec, realCPU real, realCPU imag, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size)
	{
		vec[idx].x = real;
		vec[idx].y = imag;
	}
}