#ifndef _FAME_FAST_ALGORITHMS_BIISOTROPIC_H_
#define _FAME_FAST_ALGORITHMS_BIISOTROPIC_H_

int FAME_Fast_Algorithms_Biisotropic
	(double*        Freq_array,
	 cmpx*          Ele_field_mtx,
	 CULIB_HANDLES cuHandles,
	 LAMBDAS_CUDA  Lambdas_cuda, 
	 LANCZOS_BUFFER lBuffer,
	 FFT_BUFFER    fft_buffer,
	 MTX_B 	mtx_B,
	 MATERIAL material,	 
	 int Nx, 
	 int Ny, 
	 int Nz,
	 int Nd,
	 ES  es,
	 LS  ls, 
	 string flag_CompType,
	 PROFILE* Profile);



#endif
