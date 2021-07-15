#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include <complex.h>

#include "FAME_Matrix_Vector_Production_Qr.cuh"
#include "FAME_Matrix_Vector_Production_Qrs.cuh"
#include "FAME_Matrix_Vector_Production_Bianisotropic_G_Shuffle.cuh"

#include "printDeviceArray.cuh"


void FAME_Matrix_Vector_Production_Bianisotropic_F(CULIB_HANDLES cuHandles, FFT_BUFFER fft_buffer, MTX_B mtx_B, 
         int Nx, int Ny, int Nz, int Nd, 
         cmpxGPU* Pi_Qr, cmpxGPU* Pi_Qrs, 
         cmpxGPU* Pi_Qr_110, cmpxGPU* Pi_Qrs_110,
         cmpxGPU* Pi_Qr_101, cmpxGPU* Pi_Qrs_101, 
         cmpxGPU* Pi_Qr_011, cmpxGPU* Pi_Qrs_011, 
         cmpxGPU* D_k, cmpxGPU* D_ks, 
         cmpxGPU* x, cmpxGPU* y)
{
    int size = 2 * Nd;
    int size_temp = 3 * Nx * Ny * Nz;
    size_t memsize = 8*size_temp*sizeof(cmpxGPU);

    cmpxGPU* temp_vec;
    cudaMalloc((void**)&temp_vec, memsize);

    FAME_Matrix_Vector_Production_Qr(temp_vec+0*size_temp, x+0*size, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_k, Pi_Qr);	
    FAME_Matrix_Vector_Production_Qr(temp_vec+1*size_temp, x+1*size, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_k, Pi_Qr_110);
    FAME_Matrix_Vector_Production_Qr(temp_vec+2*size_temp, x+2*size, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_k, Pi_Qr_101);
    FAME_Matrix_Vector_Production_Qr(temp_vec+3*size_temp, x+3*size, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_k, Pi_Qr_011);

    FAME_Matrix_Vector_Production_Qr(temp_vec+4*size_temp, x+4*size, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_k, Pi_Qr);
    FAME_Matrix_Vector_Production_Qr(temp_vec+5*size_temp, x+5*size, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_k, Pi_Qr_110);
    FAME_Matrix_Vector_Production_Qr(temp_vec+6*size_temp, x+6*size, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_k, Pi_Qr_101);
    FAME_Matrix_Vector_Production_Qr(temp_vec+7*size_temp, x+7*size, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_k, Pi_Qr_011);

    FAME_Matrix_Vector_Production_Bianisotropic_G_Shuffle(cuHandles, Nx, Ny, Nz, Nd, mtx_B.GInOut_index, mtx_B.GInOut_index_length, mtx_B.G, temp_vec);

    FAME_Matrix_Vector_Production_Qrs(y+0*size, temp_vec+0*size_temp, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_ks, Pi_Qrs);
    FAME_Matrix_Vector_Production_Qrs(y+1*size, temp_vec+1*size_temp, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_ks, Pi_Qrs_110);
    FAME_Matrix_Vector_Production_Qrs(y+2*size, temp_vec+2*size_temp, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_ks, Pi_Qrs_101);
    FAME_Matrix_Vector_Production_Qrs(y+3*size, temp_vec+3*size_temp, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_ks, Pi_Qrs_011);

    FAME_Matrix_Vector_Production_Qrs(y+4*size, temp_vec+4*size_temp, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_ks, Pi_Qrs);
    FAME_Matrix_Vector_Production_Qrs(y+5*size, temp_vec+5*size_temp, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_ks, Pi_Qrs_110);
    FAME_Matrix_Vector_Production_Qrs(y+6*size, temp_vec+6*size_temp, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_ks, Pi_Qrs_101);
    FAME_Matrix_Vector_Production_Qrs(y+7*size, temp_vec+7*size_temp, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_ks, Pi_Qrs_011);

    cudaFree( temp_vec );
}


void FAME_Matrix_Vector_Production_Bianisotropic_F(CULIB_HANDLES cuHandles, FFT_BUFFER fft_buffer, MTX_B mtx_B,
    int Nx, int Ny, int Nz, int Nd, 
    cmpxGPU* Pi_Qr, cmpxGPU* Pi_Qrs, 
    cmpxGPU* Pi_Qr_110, cmpxGPU* Pi_Qrs_110,
    cmpxGPU* Pi_Qr_101, cmpxGPU* Pi_Qrs_101, 
    cmpxGPU* Pi_Qr_011, cmpxGPU* Pi_Qrs_011, 
    cmpxGPU* D_kx, cmpxGPU* D_ky, cmpxGPU* D_kz,  
    cmpxGPU* x, cmpxGPU* y)
{
    int size = 2 * Nd;
    int size_temp = 3 * Nx * Ny * Nz;
    size_t memsize = 8 * size_temp * sizeof(cmpxGPU);

    cmpxGPU* temp_vec;
    cudaMalloc((void**)&temp_vec, memsize);

    FAME_Matrix_Vector_Production_Qr(temp_vec+0*size_temp, x+0*size, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qr);	
    FAME_Matrix_Vector_Production_Qr(temp_vec+1*size_temp, x+1*size, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qr_110);
    FAME_Matrix_Vector_Production_Qr(temp_vec+2*size_temp, x+2*size, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qr_101);
    FAME_Matrix_Vector_Production_Qr(temp_vec+3*size_temp, x+3*size, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qr_011);

    FAME_Matrix_Vector_Production_Qr(temp_vec+4*size_temp, x+4*size, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qr);
    FAME_Matrix_Vector_Production_Qr(temp_vec+5*size_temp, x+5*size, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qr_110);
    FAME_Matrix_Vector_Production_Qr(temp_vec+6*size_temp, x+6*size, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qr_101);
    FAME_Matrix_Vector_Production_Qr(temp_vec+7*size_temp, x+7*size, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qr_011);

    FAME_Matrix_Vector_Production_Bianisotropic_G_Shuffle(cuHandles, Nx, Ny, Nz, Nd, mtx_B.GInOut_index, mtx_B.GInOut_index_length, mtx_B.G, temp_vec);
    
    FAME_Matrix_Vector_Production_Qrs(y+0*size, temp_vec+0*size_temp, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qrs);
    FAME_Matrix_Vector_Production_Qrs(y+1*size, temp_vec+1*size_temp, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qrs_110);
    FAME_Matrix_Vector_Production_Qrs(y+2*size, temp_vec+2*size_temp, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qrs_101);
    FAME_Matrix_Vector_Production_Qrs(y+3*size, temp_vec+3*size_temp, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qrs_011);

    FAME_Matrix_Vector_Production_Qrs(y+4*size, temp_vec+4*size_temp, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qrs);
    FAME_Matrix_Vector_Production_Qrs(y+5*size, temp_vec+5*size_temp, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qrs_110);
    FAME_Matrix_Vector_Production_Qrs(y+6*size, temp_vec+6*size_temp, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qrs_101);
    FAME_Matrix_Vector_Production_Qrs(y+7*size, temp_vec+7*size_temp, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qrs_011);

    cudaFree( temp_vec );

}
