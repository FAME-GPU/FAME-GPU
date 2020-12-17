#ifndef _CG_BIISO_H_
#define _CG_BIISO_H_

int CG_Biiso
	(	CULIB_HANDLES cuHandles, 
		FFT_BUFFER 	fft_buffer,
		MTX_B            mtx_B,
		double Tol, 
		int Maxit, 
		cuDoubleComplex* rhs, 
		int Nx, int Ny, int Nz, int Nd, 
		cuDoubleComplex* D_k, 
		cuDoubleComplex* D_ks, 
		cuDoubleComplex* Pi_Qr,
		cuDoubleComplex* Pi_Pr,
		cuDoubleComplex* Pi_Qrs,		
		cuDoubleComplex* Pi_Prs,
		cuDoubleComplex* vec_y,
		PROFILE* Profile);

		int CG_Biiso
		(	CULIB_HANDLES cuHandles, 
			FFT_BUFFER fft_buffer,
			MTX_B            mtx_B,
			double Tol, 
			int Maxit, 
			cuDoubleComplex* b, 
			int Nx, 
			int Ny, 
			int Nz, 
			int Nd, 
			cuDoubleComplex* D_kx, 
			cuDoubleComplex* D_ky, 
			cuDoubleComplex* D_kz, 
			cuDoubleComplex* Pi_Qr,
			cuDoubleComplex* Pi_Pr,
			cuDoubleComplex* Pi_Qrs,		
			cuDoubleComplex* Pi_Prs,
			cuDoubleComplex* vec_y,
			PROFILE* Profile);
#endif
