#ifndef _FAME_MATRIX_VECTOR_PRODUCTION_QRS_H_
#define _FAME_MATRIX_VECTOR_PRODUCTION_QRS_H_

int FAME_Matrix_Vector_Production_Qrs(
	cmpxGPU* vec_y,
	cmpxGPU* vec_x,
	CULIB_HANDLES    cuHandles, 
	FFT_BUFFER       fft_buffer, 
	int Nx, int Ny, int Nz, int Nd,
	cmpxGPU* D_ks, 
	cmpxGPU* Pi_Qrs);

int FAME_Matrix_Vector_Production_Qrs(
	cmpxGPU* vec_y,
	cmpxGPU* vec_x,
	CULIB_HANDLES    cuHandles, 
	FFT_BUFFER       fft_buffer, 
	int Nx, int Ny, int Nz, int Nd,
	cmpxGPU* D_kx,
	cmpxGPU* D_ky,
	cmpxGPU* D_kz, 
	cmpxGPU* Pi_Qrs);

#endif
