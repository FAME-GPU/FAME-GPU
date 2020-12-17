#ifndef _LANCZOS_DECOMP_BIISOTROPIC_H_
#define _LANCZOS_DECOMP_BIISOTROPIC_H_

int Lanczos_decomp_Biisotropic(
    CULIB_HANDLES cuHandles, 
    FFT_BUFFER    fft_buffer,
    cuDoubleComplex* U, 
    MTX_B mtx_B,
    int Nx,
    int Ny,
    int Nz,
    int Nd, 
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
    double *T0,
    double *T1, 
    string flag_CompType,
    int loop_start,
    int loop_end,
    PROFILE* Profile);

#endif