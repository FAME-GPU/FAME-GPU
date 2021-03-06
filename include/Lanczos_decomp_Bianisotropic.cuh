#ifndef _LANCZOS_DECOMP_BIISOTROPIC_H_
#define _LANCZOS_DECOMP_BIISOTROPIC_H_

int Lanczos_decomp_Bianisotropic(
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
    cmpxGPU* Pi_Qr_110,
    cmpxGPU* Pi_Pr_110,
    cmpxGPU* Pi_Qrs_110,
    cmpxGPU* Pi_Prs_110,
    cmpxGPU* Pi_Qr_101,
    cmpxGPU* Pi_Pr_101,
    cmpxGPU* Pi_Qrs_101,
    cmpxGPU* Pi_Prs_101,
    cmpxGPU* Pi_Qr_011,
    cmpxGPU* Pi_Pr_011,
    cmpxGPU* Pi_Qrs_011,
    cmpxGPU* Pi_Prs_011,
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