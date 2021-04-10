#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include "printDeviceArray.cuh"
#include <complex.h>
/*
size of vec_x = 6*Nd
Update  : yilin
Date    : 2019/02/17
*/
static __global__ void dot_product_3size( int size, cmpxGPU* array_1, cmpxGPU* array_2, cmpxGPU* output );
static __global__ void dot_product_3size( int size, cmpxGPU* array_1, realGPU* array_2, cmpxGPU* output );
int FAME_Matrix_Vector_Production_invB_Biisotropic( CULIB_HANDLES cuHandles,
													MTX_B mtx_B,
													int N,
													cmpxGPU* vec_x,
													cmpxGPU* vec_y)
{
		int N_3 = 3*N; //3*Nx*Ny*Nz
		dim3 DimBlock(BLOCK_SIZE, 1, 1);
		dim3 DimGrid((N-1)/BLOCK_SIZE+1, 1, 1);
		
		cublasStatus_t cublasStatus;

		cmpxGPU one; one.x = 1.0; one.y = 0.0;
		cmpxGPU m_one_i; m_one_i.x = 0.0; m_one_i.y = -1.0;
		realGPU m_one = -1.0;

		cmpxGPU* temp;
		checkCudaErrors(cudaMalloc((void**)&temp, 3*N*sizeof(cmpxGPU)));
		
		//////// create vec_y_ele
		dot_product_3size<<<DimGrid, DimBlock>>>(N, mtx_B.B_xi, vec_x, temp );

		dot_product_3size<<<DimGrid, DimBlock>>>(N, vec_x+N_3, mtx_B.B_mu, vec_y );

		cublasStatus = PC_cublas_axpy( cuHandles.cublas_handle, N_3, &one, vec_y, 1, temp, 1 );
		assert( cublasStatus == CUBLAS_STATUS_SUCCESS );
		cublasStatus = PC_cublas_dscal( cuHandles.cublas_handle, N_3, &m_one, temp, 1 );
		assert( cublasStatus == CUBLAS_STATUS_SUCCESS );
		dot_product_3size<<<DimGrid, DimBlock>>>(N, temp, mtx_B.invPhi, vec_y );

		/////// create vec_y_mag
		dot_product_3size<<<DimGrid, DimBlock>>>(N, vec_x, mtx_B.B_eps, temp );
		
		dot_product_3size<<<DimGrid, DimBlock>>>(N, mtx_B.B_zeta, vec_x+N_3, vec_y+N_3);
		
		cublasStatus = PC_cublas_axpy( cuHandles.cublas_handle, N_3, &one, vec_y+N_3, 1, temp, 1 );
		assert( cublasStatus == CUBLAS_STATUS_SUCCESS );
		dot_product_3size<<<DimGrid, DimBlock>>>(N, temp, mtx_B.invPhi, vec_y+N_3 );

		///////
		cublasStatus = PC_cublas_scal( cuHandles.cublas_handle, 6*N, &m_one_i, vec_y, 1 );
		assert( cublasStatus == CUBLAS_STATUS_SUCCESS );
	return 0;
}





static __global__ void dot_product_3size( int size, cmpxGPU* array_1, cmpxGPU* array_2, cmpxGPU* output )
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<size)
	{
		output[idx*3].x = array_1[idx*3].x*array_2[idx*3].x - array_1[idx*3].y*array_2[idx*3].y;
		output[idx*3].y = array_1[idx*3].x*array_2[idx*3].y + array_1[idx*3].y*array_2[idx*3].x;
	
		output[idx*3+1].x = array_1[idx*3+1].x*array_2[idx*3+1].x - array_1[idx*3+1].y*array_2[idx*3+1].y;
        output[idx*3+1].y = array_1[idx*3+1].x*array_2[idx*3+1].y + array_1[idx*3+1].y*array_2[idx*3+1].x;
	
		output[idx*3+2].x = array_1[idx*3+2].x*array_2[idx*3+2].x - array_1[idx*3+2].y*array_2[idx*3+2].y;
        output[idx*3+2].y = array_1[idx*3+2].x*array_2[idx*3+2].y + array_1[idx*3+2].y*array_2[idx*3+2].x;

	}
}


static __global__ void dot_product_3size( int size, cmpxGPU* array_1, realGPU* array_2, cmpxGPU* output )
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size)
    {
        output[idx*3].x = array_1[idx*3].x*array_2[idx*3];
        output[idx*3].y = array_1[idx*3].y*array_2[idx*3];

        output[idx*3+1].x = array_1[idx*3+1].x*array_2[idx*3+1];
        output[idx*3+1].y = array_1[idx*3+1].y*array_2[idx*3+1];

        output[idx*3+2].x = array_1[idx*3+2].x*array_2[idx*3+2];
        output[idx*3+2].y = array_1[idx*3+2].y*array_2[idx*3+2];

    }
}

