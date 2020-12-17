#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include <complex.h>

#include "FAME_Matrix_Vector_Production_Pr.cuh"
#include "FAME_Matrix_Vector_Production_Qr.cuh"
#include "FAME_Matrix_Vector_Production_invB_Biisotropic.cuh"
#include "Lanczos_Biisotropic.cuh"
#include "printDeviceArray.cuh"

//#define BLOCK_SIZE 1024

int Eigen_Restoration_Biisotropic(  CULIB_HANDLES cuHandles,
                                    LAMBDAS_CUDA Lambdas_cuda,
                                    FFT_BUFFER    fft_buffer,
                                    cuDoubleComplex* Input_eigvec_mat,
                                    cuDoubleComplex* Output_eigvec_mat,
                                    MTX_B  mtx_B,
                                    int N_eig_wanted,
                                    int Nx,
                                    int Ny,
                                    int Nz,
                                    int Nd,
                                    std::string flag_CompType, PROFILE* Profile);
 
static __global__ void scaling(int size, cuDoubleComplex* array, double norm_vec);
static __global__ void ones(int size, cuDoubleComplex* array);
static __global__ void initialize(cuDoubleComplex* vec, double real, double imag, int size);

int FAME_Fast_Algorithms_Biisotropic
	(double*        Freq_array,
	 cmpx*          Ele_field_mtx,
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
	cout << "IN FAME_Fast_Algorithms_Biisotropic " << endl;

	int N = Nx*Ny*Nz;
	int Nd4 = Nd * 4;
	int N6 = 6*N;
	dim3 DimBlock(BLOCK_SIZE, 1, 1);
	dim3 DimGrid((N-1)/BLOCK_SIZE+1, 1, 1);
	size_t memsize;
	int eigen_wanted = es.nwant;
	cublasStatus_t cublasStatus;

	memsize = Nd4 * (es.nstep + 1) * sizeof(cuDoubleComplex);
	checkCudaErrors(cudaMalloc((void**) &lBuffer.dU, memsize));
	
	cuDoubleComplex* DEV_Back;
	checkCudaErrors(cudaMalloc((void**)&DEV_Back,   sizeof(cuDoubleComplex)*6*N*eigen_wanted));
	cuDoubleComplex* DEV;
	checkCudaErrors(cudaMalloc((void**)&DEV,        sizeof(cuDoubleComplex)*4*Nd*(eigen_wanted+2)*2));
	cuDoubleComplex* EW = (cuDoubleComplex*) malloc( eigen_wanted*sizeof(cuDoubleComplex));


	memsize = Nd4 * sizeof(cuDoubleComplex);
    checkCudaErrors(cudaMalloc((void**)&cuHandles.Nd2_temp1, memsize));
    checkCudaErrors(cudaMalloc((void**)&cuHandles.Nd2_temp2, memsize));
    checkCudaErrors(cudaMalloc((void**)&cuHandles.Nd2_temp3, memsize));
	checkCudaErrors(cudaMalloc((void**)&cuHandles.Nd2_temp4, memsize));

	
	if (material.chirality_in[0] > sqrt(13))
	{
		cout<<"Chirality_in > sqrt(13), indefinite matrix ！"<<endl;
		assert(0);
	}
	else if (material.chirality_in[0] < sqrt(13))
	{
	
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
	}

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
				cublasStatus=cublasZswap_v2(cuHandles.cublas_handle, 6*N, DEV_Back + i * 6*N, 1, DEV_Back + (i - 2) * 6*N, 1);
				assert( cublasStatus == CUBLAS_STATUS_SUCCESS );
				Freq_array[i] = Freq_array[i - 2];
			}

			Freq_array[0] = 0.0;
			Freq_array[1] = 0.0;

			double temp = 1.0 / sqrt(N6);
			initialize<<<DimGrid, DimBlock>>>(DEV_Back,       temp, 0.0, N);
			initialize<<<DimGrid, DimBlock>>>(DEV_Back + N6,  temp, 0.0, N);


	}

	checkCudaErrors(cudaMemcpy(Ele_field_mtx, DEV_Back, N6 * eigen_wanted * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

	cudaFree(DEV); cudaFree(DEV_Back);
	cudaFree(cuHandles.Nd2_temp1); cudaFree(cuHandles.Nd2_temp2); cudaFree(cuHandles.Nd2_temp3); cudaFree(cuHandles.Nd2_temp4);
	return 0;
}

int Eigen_Restoration_Biisotropic( 	CULIB_HANDLES cuHandles,
									LAMBDAS_CUDA Lambdas_cuda,
									FFT_BUFFER    fft_buffer,
									cuDoubleComplex* Input_eigvec_mat, 
									cuDoubleComplex* Output_eigvec_mat,
									MTX_B  mtx_B,	
									int N_eig_wanted, 
									int Nx,
									int Ny,
									int Nz,
									int Nd,
									string flag_CompType,PROFILE* Profile)
{
	int N = Nx*Ny*Nz;
	double norm_vec = 0.0;
	dim3 DimGrid(BLOCK_SIZE, 1, 1);
	dim3 DimBlock((N-1)/BLOCK_SIZE+1, 1, 1);
	cublasStatus_t cublasStatus;

	cuDoubleComplex* vec_y;
	checkCudaErrors(cudaMalloc((void**)&vec_y, 6*N*sizeof(cuDoubleComplex)));

	// Start to restore the eigenvectors

	for(int ii = 0; ii<N_eig_wanted; ii++)
	{
		if( flag_CompType == "Simple" ){
			FAME_Matrix_Vector_Production_Pr( 	cuHandles, 
												fft_buffer, 
												Input_eigvec_mat+4*Nd*ii, 
												Nx, 
												Ny,
												Nz,
												Nd,
												Lambdas_cuda.dD_k,
												Lambdas_cuda.dPi_Pr,
												vec_y);
			
			FAME_Matrix_Vector_Production_Qr( 	vec_y+3*N,
												Input_eigvec_mat+2*Nd+4*Nd*ii,				
												cuHandles, 
												fft_buffer,
												Lambdas_cuda.dD_k,
												Lambdas_cuda.dPi_Qr,
												Nx, Ny, Nz, Nd, Profile );

		}else if( flag_CompType == "General" ){
			//printDeviceArray( Input_eigvec_mat, 2*Nd, "print_Input_eigvec_mat.txt");
			FAME_Matrix_Vector_Production_Pr(	cuHandles, 
												fft_buffer,
												Input_eigvec_mat+4*Nd*ii,
												Nx,
												Ny,
												Nz,
												Nd,
												Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz,
												Lambdas_cuda.dPi_Pr,
												vec_y);									
			//printDeviceArray( vec_y, 3*N, "print_vec_y.txt");												
			FAME_Matrix_Vector_Production_Qr(   vec_y+3*N,
												Input_eigvec_mat+2*Nd+4*Nd*ii,
												cuHandles,
                                                fft_buffer,
                                                Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, 
                                                Lambdas_cuda.dPi_Qr,
                                                Nx, Ny, Nz, Nd, Profile);
		}
	
	FAME_Matrix_Vector_Production_invB_Biisotropic( cuHandles,
                                                    mtx_B,
                                                    N,
                                                    vec_y,
                                                    Output_eigvec_mat+6*N*ii);
	
	//Normalize the eigenvector
	cublasStatus=cublasDznrm2(cuHandles.cublas_handle, 6*N, Output_eigvec_mat+6*N*ii, 1, &norm_vec );
	assert( cublasStatus == CUBLAS_STATUS_SUCCESS );
	scaling<<<DimGrid, DimBlock>>>(N, Output_eigvec_mat+6*N*ii, norm_vec);

	}

	cudaFree(vec_y);
	return 0;
}



static __global__ void scaling(int size, cuDoubleComplex* array, double norm_vec)
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

static __global__ void ones(int size, cuDoubleComplex* array)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;	
	if( idx < size )
	{	for( int i=0; i<6; i++)
		{
			array[idx*6+i].x = 1.0; array[idx*6+i].y = 0.0;
		}
	}

}

static __global__ void initialize(cuDoubleComplex* vec, double real, double imag, int size)
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











