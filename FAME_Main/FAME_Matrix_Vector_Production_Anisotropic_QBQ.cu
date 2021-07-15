#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include "FAME_Matrix_Vector_Production_Anisotropic_N_Shuffle.cuh"
#include "FAME_Matrix_Vector_Production_Qr.cuh"
#include "FAME_Matrix_Vector_Production_Qrs.cuh"

int FAME_Matrix_Vector_Production_Anisotropic_QBQ(
	cmpxGPU* vec_y,
	cmpxGPU* vec_x,
	CULIB_HANDLES    cuHandles,
	FFT_BUFFER       fft_buffer,
	MTX_B            mtx_B,
	cmpxGPU* D_k,
	cmpxGPU* D_ks,
	cmpxGPU* Pi_Qr,
	cmpxGPU* Pi_Qrs,
	cmpxGPU* Pi_Qr_110,
    cmpxGPU* Pi_Qrs_110,
    cmpxGPU* Pi_Qr_101,
    cmpxGPU* Pi_Qrs_101,
    cmpxGPU* Pi_Qr_011,
    cmpxGPU* Pi_Qrs_011,
	int Nx, int Ny, int Nz, int Nd)
{
    int N = Nx * Ny * Nz;
    int N3 = N * 3;
	int Nd2 = Nd * 2;

	cmpxGPU* vec_y_1 = cuHandles.temp_vec1;

    FAME_Matrix_Vector_Production_Qr(vec_y_1,          vec_x,           cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_k, Pi_Qr);
	FAME_Matrix_Vector_Production_Qr(vec_y_1 + N3,     vec_x + Nd2,     cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_k, Pi_Qr_110);
	FAME_Matrix_Vector_Production_Qr(vec_y_1 + 2 * N3, vec_x + 2 * Nd2, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_k, Pi_Qr_101);
	FAME_Matrix_Vector_Production_Qr(vec_y_1 + 3 * N3, vec_x + 3 * Nd2, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_k, Pi_Qr_011);
 
    FAME_Matrix_Vector_Production_Anisotropic_N_Shuffle(cuHandles, Nx, Ny, Nz, Nd, mtx_B.GInOut_index, mtx_B.GInOut_index_length, mtx_B.N, vec_y_1);

    FAME_Matrix_Vector_Production_Qrs(vec_y,           vec_y_1,          cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_ks, Pi_Qrs);
	FAME_Matrix_Vector_Production_Qrs(vec_y + Nd2,     vec_y_1 + N3,     cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_ks, Pi_Qrs_110);
	FAME_Matrix_Vector_Production_Qrs(vec_y + 2 * Nd2, vec_y_1 + 2 * N3, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_ks, Pi_Qrs_101);
	FAME_Matrix_Vector_Production_Qrs(vec_y + 3 * Nd2, vec_y_1 + 3 * N3, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_ks, Pi_Qrs_011);

	return 0;
}

int FAME_Matrix_Vector_Production_Anisotropic_QBQ(
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
	cmpxGPU* Pi_Qr_110,
    cmpxGPU* Pi_Qrs_110,
    cmpxGPU* Pi_Qr_101,
    cmpxGPU* Pi_Qrs_101,
    cmpxGPU* Pi_Qr_011,
    cmpxGPU* Pi_Qrs_011,
	int Nx, int Ny, int Nz, int Nd)
{
    int N = Nx * Ny * Nz;
    int N3 = N * 3;
	int Nd2 = Nd * 2;

	cmpxGPU* vec_y_1 = cuHandles.temp_vec1;

    FAME_Matrix_Vector_Production_Qr(vec_y_1,          vec_x,           cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qr);
	FAME_Matrix_Vector_Production_Qr(vec_y_1 + N3,     vec_x + Nd2,     cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qr_110);
	FAME_Matrix_Vector_Production_Qr(vec_y_1 + 2 * N3, vec_x + 2 * Nd2, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qr_101);
	FAME_Matrix_Vector_Production_Qr(vec_y_1 + 3 * N3, vec_x + 3 * Nd2, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qr_011);
 
    FAME_Matrix_Vector_Production_Anisotropic_N_Shuffle(cuHandles, Nx, Ny, Nz, Nd, mtx_B.GInOut_index, mtx_B.GInOut_index_length, mtx_B.N, vec_y_1);

    FAME_Matrix_Vector_Production_Qrs(vec_y,           vec_y_1,          cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qrs);
	FAME_Matrix_Vector_Production_Qrs(vec_y + Nd2,     vec_y_1 + N3,     cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qrs_110);
	FAME_Matrix_Vector_Production_Qrs(vec_y + 2 * Nd2, vec_y_1 + 2 * N3, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qrs_101);
	FAME_Matrix_Vector_Production_Qrs(vec_y + 3 * Nd2, vec_y_1 + 3 * N3, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz, Pi_Qrs_011);


    return 0;
}



