#ifndef _LANCZOS_DECOMP_BIISOTROPIC_H_
#define _LANCZOS_DECOMP_BIISOTROPIC_H_

int Lanczos_decomp_Biisotropic(
    CULIB_HANDLES cuHandles, 
    FFT_BUFFER    fft_buffer,
    cmpxGPU* U, 
    MTX_B mtx_B,
    int Nx,
    int Ny,
    int Nz,
    int Nd, 
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
    realGPU *T0,
    realGPU *T1, 
    string flag_CompType,
    int loop_start,
    int loop_end,
    PROFILE* Profile);

#endif