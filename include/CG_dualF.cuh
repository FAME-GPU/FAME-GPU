#ifndef _CG_DUALF_H_
#define _CG_DUALF_H_

int CG_dualF(cmpxGPU *b,
    cmpxGPU *vec_y,
    CULIB_HANDLES cuHandles, 
    FFT_BUFFER    fft_buffer,
    MTX_B mtx_B,
    int Nx,
    int Ny,
    int Nz,
    int Nd,
    int max_iter,
    realGPU tol,
    cmpxGPU* Pi_Pr,
    cmpxGPU* Pi_Prs,
    cmpxGPU* Pi_Pr_110,
    cmpxGPU* Pi_Prs_110,
    cmpxGPU* Pi_Pr_101,
    cmpxGPU* Pi_Prs_101,
    cmpxGPU* Pi_Pr_011,
    cmpxGPU* Pi_Prs_011,
    cmpxGPU* D_k,
    cmpxGPU* D_ks);

int CG_dualF(cmpxGPU *b,
    cmpxGPU *vec_y,
    CULIB_HANDLES cuHandles, 
    FFT_BUFFER    fft_buffer,
    MTX_B mtx_B,
    int Nx,
    int Ny,
    int Nz,
    int Nd,
    int max_iter,
    realGPU tol,
    cmpxGPU* Pi_Pr,
    cmpxGPU* Pi_Prs,
    cmpxGPU* Pi_Pr_110,
    cmpxGPU* Pi_Prs_110,
    cmpxGPU* Pi_Pr_101,
    cmpxGPU* Pi_Prs_101,
    cmpxGPU* Pi_Pr_011,
    cmpxGPU* Pi_Prs_011,
    cmpxGPU* D_kx,
    cmpxGPU* D_ky,
    cmpxGPU* D_kz);

#endif