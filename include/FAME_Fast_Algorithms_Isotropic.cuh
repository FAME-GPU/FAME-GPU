#ifndef _FAME_FAST_ALGORITHMS_ISOTROPIC_H_
#define _FAME_FAST_ALGORITHMS_ISOTROPIC_H_

int FAME_Fast_Algorithms_Isotropic(
	double*        Freq_array,
	cmpx*          Ele_field_mtx,
	CULIB_HANDLES  cuHandles,
	LANCZOS_BUFFER lBuffer,
	FFT_BUFFER     fft_buffer,
	LAMBDAS_CUDA   Lambdas_cuda,
	MTX_B          mtx_B,
	ES 			   es,
	LS 			   ls,
	int Nx, int Ny, int Nz, int Nd, int N,
	string flag_CompType, PROFILE* Profile);
	
#endif
