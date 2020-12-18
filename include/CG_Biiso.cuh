#ifndef _CG_BIISO_H_
#define _CG_BIISO_H_

int CG_Biiso
	(	CULIB_HANDLES cuHandles, 
		FFT_BUFFER 	fft_buffer,
		MTX_B            mtx_B,
		realGPU Tol, 
		int Maxit, 
		cmpxGPU* rhs, 
		int Nx, int Ny, int Nz, int Nd, 
		cmpxGPU* D_k, 
		cmpxGPU* D_ks, 
		cmpxGPU* Pi_Qr,
		cmpxGPU* Pi_Pr,
		cmpxGPU* Pi_Qrs,		
		cmpxGPU* Pi_Prs,
		cmpxGPU* vec_y,
		PROFILE* Profile);

		int CG_Biiso
		(	CULIB_HANDLES cuHandles, 
			FFT_BUFFER fft_buffer,
			MTX_B            mtx_B,
			realGPU Tol, 
			int Maxit, 
			cmpxGPU* b, 
			int Nx, 
			int Ny, 
			int Nz, 
			int Nd, 
			cmpxGPU* D_kx, 
			cmpxGPU* D_ky, 
			cmpxGPU* D_kz, 
			cmpxGPU* Pi_Qr,
			cmpxGPU* Pi_Pr,
			cmpxGPU* Pi_Qrs,		
			cmpxGPU* Pi_Prs,
			cmpxGPU* vec_y,
			PROFILE* Profile);
#endif
