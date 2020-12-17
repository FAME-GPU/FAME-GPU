#ifndef _FAME_MATRIX_VECTOR_PRODUCTION_QR_H_
#define _FAME_MATRIX_VECTOR_PRODUCTION_QR_H_

int FAME_Matrix_Vector_Production_Qr(
	cuDoubleComplex* vec_y,
	cuDoubleComplex* vec_x,
	CULIB_HANDLES    cuHandles, 
	FFT_BUFFER       fft_buffer, 
	cuDoubleComplex* D_k, 
	cuDoubleComplex* Pi_Qr,
	int Nx, int Ny, int Nz, int Nd, PROFILE* Profile);

int FAME_Matrix_Vector_Production_Qr(
	cuDoubleComplex* vec_y,
	cuDoubleComplex* vec_x,
	CULIB_HANDLES    cuHandles, 
	FFT_BUFFER       fft_buffer, 
	cuDoubleComplex* D_kx, 
	cuDoubleComplex* D_ky, 
	cuDoubleComplex* D_kz, 
	cuDoubleComplex* Pi_Qr,
	int Nx, int Ny, int Nz, int Nd, PROFILE* Profile);

#endif
