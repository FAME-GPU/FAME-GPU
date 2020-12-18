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
        realGPU* Lambda_q_sqrt,
        cmpxGPU* Pi_Qr,
        cmpxGPU* Pi_Pr,
        cmpxGPU* Pi_Qrs,
        cmpxGPU* Pi_Prs,
        cmpxGPU* D_k,
        cmpxGPU* D_ks,
		cmpxGPU* D_kx,
		cmpxGPU* D_ky,
        cmpxGPU* D_kz,
        realGPU*          Freq_array, 
        cmpxGPU* ev,
        string flag_CompType,
        PROFILE* Profile);

#endif