#ifndef _FAME_MATRIX_VECTOR_PRODUCTION_PRS_H_
#define _FAME_MATRIX_VECTOR_PRODUCTION_PRS_H_

int FAME_Matrix_Vector_Production_Prs( CULIB_HANDLES cuHandles, FFT_BUFFER fft_buffer, cuDoubleComplex* vec_x, int Nx, int Ny, int Nz, int Nd, cuDoubleComplex* D_ks, cuDoubleComplex* Pi_Prs, cuDoubleComplex* vec_y);

int FAME_Matrix_Vector_Production_Prs( CULIB_HANDLES cuHandles, FFT_BUFFER fft_buffer, cuDoubleComplex* vec_x, int Nx, int Ny, int Nz, int Nd, cuDoubleComplex* D_kx, cuDoubleComplex* D_ky, cuDoubleComplex* D_kz, cuDoubleComplex* Pi_Prs, cuDoubleComplex* vec_y);

#endif
