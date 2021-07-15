#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include <complex.h>

#include "FAME_Matrix_Vector_Production_Qr.cuh"
#include "FAME_Matrix_Vector_Production_Qrs.cuh"
#include "FAME_Matrix_Vector_Production_Pr.cuh"
#include "FAME_Matrix_Vector_Production_Prs.cuh"
#include "FAME_Matrix_Vector_Production_invPhi_Biisotropic.cuh"
#include "printDeviceArray.cuh"

/*
size of vec_x = 4*Nd
size of vec_y = 4*Nd
Author  : yilin
Date    : 2018/10/30
*/
static __global__ void dot_pro_add( int size,
                                    cmpxGPU* B_zeta,
                                    cmpxGPU* vec_y_ele,
                                    cmpxGPU* vec_y_mag,
                                    cmpxGPU* ovec);
static __global__ void dot_pro_minus(   int size,
                                        cmpxGPU* B_zeta,
                                        cmpxGPU* vec_y_ele,
                                        cmpxGPU* vec_y_mag,
                                        cmpxGPU* ovec);


// (Matlab : FAME_Matrix_Vector_Production_Biisotropic_Ar_Simple
void FAME_Matrix_Vector_Production_Biisotropic_Ar
		(CULIB_HANDLES cuHandles,
		 FFT_BUFFER fft_buffer,
		 cmpxGPU* vec_x,
		 MTX_B mtx_B,
		 int Nx,
		 int Ny, 
		 int Nz, 
		 int Nd,
		 cmpxGPU* Pi_Qr,  
		 cmpxGPU* Pi_Pr,
		 cmpxGPU* Pi_Qrs, 
		 cmpxGPU* Pi_Prs,
		 cmpxGPU* D_k,
		 cmpxGPU* D_ks,
		 cmpxGPU* vec_y,
		 PROFILE* Profile)
{
	dim3 DimBlock(BLOCK_SIZE,1,1);
    dim3 DimGrid((Nd-1)/BLOCK_SIZE +1,1,1);
	cublasStatus_t cublasStatus;

	cmpxGPU* temp_vec_y_ele;
	cmpxGPU* temp_vec_y_mag;
	cmpxGPU* vec_y_ele;
	cmpxGPU* vec_y_mag;


	int N=Nx*Ny*Nz;
	int size = 3*N;
	int memsize = size*sizeof(cmpxGPU);
	cudaMalloc((void**)&vec_y_ele, memsize);
	cudaMalloc((void**)&vec_y_mag, memsize);	
	cudaMalloc((void**)&temp_vec_y_ele, memsize);
	cudaMalloc((void**)&temp_vec_y_mag, memsize);


	FAME_Matrix_Vector_Production_Pr( vec_y_ele, vec_x, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_k, Pi_Pr);

	FAME_Matrix_Vector_Production_Qr( vec_y_mag, vec_x+2*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_k, Pi_Qr);
	
	// temp_vec_y_ele = B.B_zeta_s.*vec_y_ele + vec_y_mag;	
	dot_pro_add<<<DimGrid, DimBlock>>>( N, mtx_B.B_zeta_s, vec_y_ele, vec_y_mag, temp_vec_y_ele); 
	
	// temp_vec_y_mag = -vec_y_ele;
	realGPU alpha = -1.0;
	cublasStatus = FAME_cublas_dscal(cuHandles.cublas_handle, size, &alpha, vec_y_ele, 1); 
	assert( cublasStatus == CUBLAS_STATUS_SUCCESS );
	
	checkCudaErrors(cudaMemcpy(vec_y_mag, vec_y_ele, memsize, cudaMemcpyDeviceToDevice));

	//invPhi
	//dot_pro<<<DimGrid, DimBlock>>>( size, mtx_B.invPhi, temp_vec_y_ele, vec_y_ele);
	FAME_Matrix_Vector_Production_invPhi_Biisotropic(cuHandles, size, mtx_B.invPhi, temp_vec_y_ele, vec_y_ele);

	// temp_vec_y_ele = B.B_zeta.*vec_y_ele - vec_y_mag;
	dot_pro_minus<<<DimGrid, DimBlock>>>( N, mtx_B.B_zeta, vec_y_ele, vec_y_mag, temp_vec_y_ele);
		
	FAME_Matrix_Vector_Production_Prs(vec_y, temp_vec_y_ele, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_ks, Pi_Prs);
	FAME_Matrix_Vector_Production_Qrs(vec_y+2*Nd, vec_y_ele,  cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_ks, Pi_Qrs);	

	cudaFree( temp_vec_y_ele );
	cudaFree( temp_vec_y_mag );
	cudaFree( vec_y_ele );
	cudaFree( vec_y_mag );

}


void FAME_Matrix_Vector_Production_Biisotropic_Ar
        (CULIB_HANDLES cuHandles,
		 FFT_BUFFER fft_buffer,
         cmpxGPU* vec_x,
         MTX_B mtx_B,
         int Nx,
         int Ny,
         int Nz,
         int Nd,
         cmpxGPU* Pi_Qr,
         cmpxGPU* Pi_Pr,
         cmpxGPU* Pi_Qrs,
		 cmpxGPU* Pi_Prs,
		 cmpxGPU* D_kx,
         cmpxGPU* D_ky,
         cmpxGPU* D_kz,
         cmpxGPU* vec_y, PROFILE* Profile)
{
	dim3 DimBlock(BLOCK_SIZE,1,1);
    dim3 DimGrid((Nd-1)/BLOCK_SIZE +1,1,1);
	cublasStatus_t cublasStatus;

	cmpxGPU* temp_vec_y_ele;
    cmpxGPU* temp_vec_y_mag;
    cmpxGPU* vec_y_ele;
    cmpxGPU* vec_y_mag;

    //int memsize = 3*Nd*sizeof(cmpxGPU);
    
	
	int N=Nx*Ny*Nz;
	int size = 3*N;
    int memsize = size*sizeof(cmpxGPU);
    checkCudaErrors(cudaMalloc((void**)&vec_y_ele, memsize));
	checkCudaErrors(cudaMalloc((void**)&vec_y_mag, memsize));
	checkCudaErrors(cudaMalloc((void**)&temp_vec_y_ele, memsize));
	checkCudaErrors(cudaMalloc((void**)&temp_vec_y_mag, memsize));


	FAME_Matrix_Vector_Production_Pr( vec_y_ele, vec_x, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Pr);	
    FAME_Matrix_Vector_Production_Qr( vec_y_mag, vec_x+2*Nd, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qr);	


	dot_pro_add<<<DimGrid, DimBlock>>>( N, mtx_B.B_zeta_s, vec_y_ele, vec_y_mag, temp_vec_y_ele);


	realGPU alpha = -1.0;
	cublasStatus = FAME_cublas_dscal(cuHandles.cublas_handle, size, &alpha, vec_y_ele, 1);
	assert( cublasStatus == CUBLAS_STATUS_SUCCESS );
    checkCudaErrors(cudaMemcpy(vec_y_mag, vec_y_ele, memsize, cudaMemcpyDeviceToDevice));

	FAME_Matrix_Vector_Production_invPhi_Biisotropic(cuHandles, size, mtx_B.invPhi, temp_vec_y_ele, vec_y_ele);

	dot_pro_minus<<<DimGrid, DimBlock>>>( N, mtx_B.B_zeta, vec_y_ele, vec_y_mag, temp_vec_y_ele);

	
	FAME_Matrix_Vector_Production_Prs(vec_y, temp_vec_y_ele, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Prs);

    FAME_Matrix_Vector_Production_Qrs(vec_y+2*Nd, vec_y_ele, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky , D_kz, Pi_Qrs);

	cudaFree( temp_vec_y_ele );
	cudaFree( temp_vec_y_mag );
	cudaFree( vec_y_ele );
	cudaFree( vec_y_mag );

}

static __global__ void dot_pro_add(	int size,
									cmpxGPU* B_zeta_s, 
								   	cmpxGPU* vec_y_ele, 
									cmpxGPU* vec_y_mag,
									cmpxGPU* ovec)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if( idx < size )
	{
		ovec[idx*3].x = B_zeta_s[idx*3].x*vec_y_ele[idx*3].x - B_zeta_s[idx*3].y*vec_y_ele[idx*3].y + vec_y_mag[idx*3].x;
		ovec[idx*3].y = B_zeta_s[idx*3].x*vec_y_ele[idx*3].y + B_zeta_s[idx*3].y*vec_y_ele[idx*3].x + vec_y_mag[idx*3].y;

		ovec[idx*3 + 1].x = B_zeta_s[idx*3+1].x*vec_y_ele[idx*3+1].x - B_zeta_s[idx*3+1].y*vec_y_ele[idx*3+1].y + vec_y_mag[idx*3+1].x;
        ovec[idx*3 + 1].y = B_zeta_s[idx*3+1].x*vec_y_ele[idx*3+1].y + B_zeta_s[idx*3+1].y*vec_y_ele[idx*3+1].x + vec_y_mag[idx*3+1].y;
	
		ovec[idx*3 + 2].x = B_zeta_s[idx*3+2].x*vec_y_ele[idx*3+2].x - B_zeta_s[idx*3+2].y*vec_y_ele[idx*3+2].y + vec_y_mag[idx*3+2].x;
        ovec[idx*3 + 2].y = B_zeta_s[idx*3+2].x*vec_y_ele[idx*3+2].y + B_zeta_s[idx*3+2].y*vec_y_ele[idx*3+2].x + vec_y_mag[idx*3+2].y;

	}
	
}
static __global__ void dot_pro_minus( 	int size,
                                    	cmpxGPU* B_zeta,
                                    	cmpxGPU* vec_y_ele,
                                    	cmpxGPU* vec_y_mag,
                                    	cmpxGPU* ovec)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if( idx < size )
    {
        ovec[idx*3].x = B_zeta[idx*3].x*vec_y_ele[idx*3].x - B_zeta[idx*3].y*vec_y_ele[idx*3].y - vec_y_mag[idx*3].x;
        ovec[idx*3].y = B_zeta[idx*3].x*vec_y_ele[idx*3].y + B_zeta[idx*3].y*vec_y_ele[idx*3].x - vec_y_mag[idx*3].y;
		
		ovec[idx*3+1].x = B_zeta[idx*3+1].x*vec_y_ele[idx*3+1].x - B_zeta[idx*3+1].y*vec_y_ele[idx*3+1].y - vec_y_mag[idx*3+1].x;
        ovec[idx*3+1].y = B_zeta[idx*3+1].x*vec_y_ele[idx*3+1].y + B_zeta[idx*3+1].y*vec_y_ele[idx*3+1].x - vec_y_mag[idx*3+1].y;

		ovec[idx*3+2].x = B_zeta[idx*3+2].x*vec_y_ele[idx*3+2].x - B_zeta[idx*3+2].y*vec_y_ele[idx*3+2].y - vec_y_mag[idx*3+2].x;
        ovec[idx*3+2].y = B_zeta[idx*3+2].x*vec_y_ele[idx*3+2].y + B_zeta[idx*3+2].y*vec_y_ele[idx*3+2].x - vec_y_mag[idx*3+2].y;

    }

}

