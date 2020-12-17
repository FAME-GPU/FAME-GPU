#ifndef _INVLANCZOS_BIISOTROPIC_H_
#define _INVLANCZOS_BIISOTROPIC_H_

int Lanczos_Biisotropic
( 	CULIB_HANDLES cuHandles, 
    FFT_BUFFER    fft_buffer,
    LANCZOS_BUFFER lBuffer,
		MTX_B mtx_B,
        int Nx,
        int Ny,
        int Nz,
        int Nd,
        ES  es,
        LS  ls,
        double* Lambda_q_sqrt,
        cuDoubleComplex* Pi_Qr,
        cuDoubleComplex* Pi_Pr,
        cuDoubleComplex* Pi_Qrs,
        cuDoubleComplex* Pi_Prs,
        cuDoubleComplex* D_k,
        cuDoubleComplex* D_ks,
		cuDoubleComplex* D_kx,
		cuDoubleComplex* D_ky,
        cuDoubleComplex* D_kz,
        double*          Freq_array, 
        cuDoubleComplex* ev,
        string flag_CompType,
        PROFILE* Profile);

#endif