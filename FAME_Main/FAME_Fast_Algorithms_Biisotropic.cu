#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include <complex.h>

#include "FAME_Matrix_Vector_Production_Pr.cuh"
#include "FAME_Matrix_Vector_Production_Qr.cuh"
#include "FAME_Matrix_Vector_Production_invB_Biisotropic.cuh"
#include "Lanczos_Biisotropic.cuh"
#include "printDeviceArray.cuh"

int Eigen_Restoration_Biisotropic( 	CULIB_HANDLES cuHandles,
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
 
static __global__ void scaling(int size, cmpxGPU* array, realCPU norm_vec);
static __global__ void initialize(cmpxGPU* vec, realCPU real, realCPU imag, int size);

int FAME_Fast_Algorithms_Biisotropic
	(realCPU* Freq_array,
	 cmpxCPU* Ele_field_mtx,
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
	cout << "IN FAME_Fast_Algorithms_Bi-isotropic " << endl;

	int N = Nx*Ny*Nz;
	int Nd4 = Nd * 4;
	int N6 = 6*N;
	dim3 DimBlock(BLOCK_SIZE, 1, 1);
	dim3 DimGrid((N-1)/BLOCK_SIZE+1, 1, 1);
	size_t memsize;
	int eigen_wanted = es.nwant;
	cublasStatus_t cublasStatus;

	memsize = Nd4 * (es.nstep + 1) * sizeof(cmpxGPU);
	checkCudaErrors(cudaMalloc((void**) &lBuffer.dU, memsize));
	
	cmpxGPU* DEV_Back;
	checkCudaErrors(cudaMalloc((void**)&DEV_Back,   sizeof(cmpxGPU)*6*N*eigen_wanted));
	cmpxGPU* DEV;
	checkCudaErrors(cudaMalloc((void**)&DEV,        sizeof(cmpxGPU)*4*Nd*(eigen_wanted+2)*2));

	memsize = Nd4 * sizeof(cmpxGPU);

    checkCudaErrors(cudaMalloc((void**)&cuHandles.Nd2_temp2, memsize));
    checkCudaErrors(cudaMalloc((void**)&cuHandles.Nd2_temp3, memsize));
	checkCudaErrors(cudaMalloc((void**)&cuHandles.Nd2_temp4, memsize));

	
	Lanczos_Biisotropic( cuHandles,
		fft_buffer, lBuffer,
		mtx_B, Nx, Ny, Nz, Nd, es, ls,
		Lambdas_cuda.Lambda_q_sqrt,
		Lambdas_cuda.dPi_Qr,
		Lambdas_cuda.dPi_Pr,
		Lambdas_cuda.dPi_Qrs,
		Lambdas_cuda.dPi_Prs,
		Lambdas_cuda.dD_k,
		Lambdas_cuda.dD_ks,
		Lambdas_cuda.dD_kx,
		Lambdas_cuda.dD_ky,
		Lambdas_cuda.dD_kz,
		Freq_array, DEV,
		flag_CompType,
		Profile );

	cudaFree(lBuffer.dU);

	Eigen_Restoration_Biisotropic(  cuHandles, Lambdas_cuda,
		fft_buffer,
		DEV, DEV_Back,
		mtx_B, eigen_wanted,
		Nx, Ny, Nz, Nd,
		flag_CompType, Profile);

	if(Nd == N-1)
	{
			for(int i = es.nwant - 1; i >= 2 ; i--)
			{
				cublasStatus=FAME_cublas_swap(cuHandles.cublas_handle, 6*N, DEV_Back + i * 6*N, 1, DEV_Back + (i - 2) * 6*N, 1);
				assert( cublasStatus == CUBLAS_STATUS_SUCCESS );
				Freq_array[i] = Freq_array[i - 2];
			}

			Freq_array[0] = 0.0;
			Freq_array[1] = 0.0;

			realCPU temp = 1.0 / sqrt(N6);
			initialize<<<DimGrid, DimBlock>>>(DEV_Back,       temp, 0.0, N);
			initialize<<<DimGrid, DimBlock>>>(DEV_Back + N6,  temp, 0.0, N);


	}

	checkCudaErrors(cudaMemcpy(Ele_field_mtx, DEV_Back, N6 * eigen_wanted * sizeof(cmpxGPU), cudaMemcpyDeviceToHost));

	cudaFree(DEV); cudaFree(DEV_Back);
	return 0;
}

int Eigen_Restoration_Biisotropic( 	CULIB_HANDLES cuHandles,
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
									string flag_CompType, PROFILE* Profile)
{
	int N = Nx*Ny*Nz;
	realCPU norm_vec = 0.0;
	dim3 DimBlock(BLOCK_SIZE, 1, 1);
	dim3 DimGrid((N-1)/BLOCK_SIZE+1, 1, 1);
	cublasStatus_t cublasStatus;

	cmpxGPU* vec_y;
	checkCudaErrors(cudaMalloc((void**)&vec_y, 6*N*sizeof(cmpxGPU)));

	// Start to restore the eigenvectors

	for(int ii = 0; ii < N_eig_wanted; ii++)
	{
		if( flag_CompType == "Simple" )
		{
			FAME_Matrix_Vector_Production_Pr(vec_y, Input_eigvec_mat+4*Nd*ii, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, 
				Lambdas_cuda.dD_k, Lambdas_cuda.dPi_Pr);																				
			FAME_Matrix_Vector_Production_Qr(vec_y + 3 * N, Input_eigvec_mat+2*Nd+4*Nd*ii, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, 
				Lambdas_cuda.dD_k, Lambdas_cuda.dPi_Qr);

		}
		else if( flag_CompType == "General" )
		{
			FAME_Matrix_Vector_Production_Pr(vec_y, Input_eigvec_mat+4*Nd*ii, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, 
				Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Pr);																				
			FAME_Matrix_Vector_Production_Qr(vec_y + 3 * N, Input_eigvec_mat+2*Nd+4*Nd*ii, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, 
				Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Qr);
		}
	
	FAME_Matrix_Vector_Production_invB_Biisotropic( cuHandles,
                                                    mtx_B,
                                                    N,
                                                    vec_y,
                                                    Output_eigvec_mat+6*N*ii);
	
	//Normalize the eigenvector
	cublasStatus=FAME_cublas_nrm2(cuHandles.cublas_handle, 6*N, Output_eigvec_mat+6*N*ii, 1, &norm_vec );
	assert( cublasStatus == CUBLAS_STATUS_SUCCESS );
	scaling<<<DimGrid, DimBlock>>>(N, Output_eigvec_mat+6*N*ii, norm_vec);

	}

	cudaFree(vec_y);
	return 0;
}



static __global__ void scaling(int size, cmpxGPU* array, realCPU norm_vec)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size)
	{
		for( int i=0; i<6; i++)
		{
			array[idx*6+i].x = array[idx*6+i].x/norm_vec;	
			array[idx*6+i].y = array[idx*6+i].y/norm_vec;
		}
	}

}

static __global__ void initialize(cmpxGPU* vec, realCPU real, realCPU imag, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// printf("%d\t", idx);
	if( idx < size )
	{	for( int i=0; i<6; i++)
		{
			vec[idx*6+i].x = real; vec[idx*6+i].y = 0.0;
		}
	}
}











