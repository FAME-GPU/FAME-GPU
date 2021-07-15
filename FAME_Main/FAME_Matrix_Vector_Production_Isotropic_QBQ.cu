#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include "FAME_Matrix_Vector_Production_Qr.cuh"
#include "FAME_Matrix_Vector_Production_Qrs.cuh"

static __global__ void dot_product(cmpxGPU* vec_y, realGPU* invB_eps, int size);

int FAME_Matrix_Vector_Production_Isotropic_QBQ(
	cmpxGPU* vec_y,
	cmpxGPU* vec_x,
	CULIB_HANDLES    cuHandles,
	FFT_BUFFER       fft_buffer,
	MTX_B            mtx_B,
	cmpxGPU* D_k,
	cmpxGPU* D_ks,
	cmpxGPU* Pi_Qr,
	cmpxGPU* Pi_Qrs,
	int Nx, int Ny, int Nz, int Nd)
{
    int N = Nx * Ny * Nz;
    int N3 = N * 3;

	cmpxGPU* vec_y_1 = cuHandles.N3_temp1;

	dim3 DimBlock(BLOCK_SIZE, 1, 1);
    dim3 DimGrid((N3 - 1) / BLOCK_SIZE + 1, 1, 1);
 //cout<<"in QBQ SIMPLE"<<endl;   

    FAME_Matrix_Vector_Production_Qr(vec_y_1, vec_x, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_k, Pi_Qr);
 
    dot_product<<<DimGrid, DimBlock>>>(vec_y_1, mtx_B.invB_eps, N3);

    FAME_Matrix_Vector_Production_Qrs(vec_y, vec_y_1, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_ks, Pi_Qrs);

	return 0;
}

int FAME_Matrix_Vector_Production_Isotropic_QBQ(
	cmpxGPU* vec_y,
	cmpxGPU* vec_x,
	CULIB_HANDLES    cuHandles,
	FFT_BUFFER       fft_buffer,
	MTX_B            mtx_B,
	cmpxGPU* D_kx,
	cmpxGPU* D_ky,
	cmpxGPU* D_kz,
	cmpxGPU* Pi_Qr,
	cmpxGPU* Pi_Qrs,
	int Nx, int Ny, int Nz, int Nd)
{
    int N = Nx * Ny * Nz;
    int N3 = N * 3;

	cmpxGPU* vec_y_1 = cuHandles.N3_temp1;

	dim3 DimBlock(BLOCK_SIZE, 1, 1);
    dim3 DimGrid((N3 - 1) / BLOCK_SIZE + 1, 1, 1);
    

	FAME_Matrix_Vector_Production_Qr(vec_y_1, vec_x, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qr);



	dot_product<<<DimGrid, DimBlock>>>(vec_y_1, mtx_B.invB_eps, N3);

	FAME_Matrix_Vector_Production_Qrs(vec_y, vec_y_1, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qrs);


    return 0;
}

static __global__ void dot_product(cmpxGPU* vec_y, realGPU* invB_eps, int size)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < size)
	{
		vec_y[idx].x = vec_y[idx].x * invB_eps[idx];
		vec_y[idx].y = vec_y[idx].y * invB_eps[idx];
	}

}
