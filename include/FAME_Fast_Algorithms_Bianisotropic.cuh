#ifndef _FAME_FAST_ALGORITHMS_BIANISOTROPIC_H_
#define _FAME_FAST_ALGORITHMS_BIANISOTROPIC_H_

int FAME_Fast_Algorithms_Bianisotropic
	(realCPU* Freq_array,
	 cmpxCPU* Ele_field_mtx,
	 cmpxCPU* Dis_field_mtx,
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