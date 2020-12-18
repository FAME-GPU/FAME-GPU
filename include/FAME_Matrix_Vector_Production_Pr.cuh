#ifndef _FAME_MATRIX_VECTOR_PRODUCTION_PR_H_
#define _FAME_MATRIX_VECTOR_PRODUCTION_PR_H_

int FAME_Matrix_Vector_Production_Pr(CULIB_HANDLES cuHandles, FFT_BUFFER fft_buffer, cmpxGPU* vec_x, int Nx, int Ny, int Nz, int Nd, cmpxGPU* D_k, cmpxGPU* Pi_Pr, cmpxGPU* vec_y);

int FAME_Matrix_Vector_Production_Pr(CULIB_HANDLES cuHandles, FFT_BUFFER fft_buffer, cmpxGPU* vec_x, int Nx, int Ny, int Nz, int Nd, cmpxGPU* D_kx, cmpxGPU* D_ky, cmpxGPU* D_kz, cmpxGPU* Pi_Pr, cmpxGPU* vec_y);

#endif
