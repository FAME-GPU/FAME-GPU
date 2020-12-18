#ifndef _CG_H_
#define _CG_H_
int CG(
    cmpxGPU* vec_y,
    cmpxGPU* b,
    CULIB_HANDLES    cuHandles,
    FFT_BUFFER       fft_buffer,
    MTX_B            mtx_B,
    cmpxGPU* D_k,
    cmpxGPU* D_ks,
    cmpxGPU* Pi_Qr,
    cmpxGPU* Pi_Qrs,
    int Nx, int Ny, int Nz, int Nd,
    int Maxit, realGPU Tol,
    PROFILE* Profile);

int CG(
    cmpxGPU* vec_y,
    cmpxGPU* b,
    CULIB_HANDLES    cuHandles,
    FFT_BUFFER       fft_buffer,
    MTX_B            mtx_B,
    cmpxGPU* D_kx,
    cmpxGPU* D_ky,
    cmpxGPU* D_kz,
    cmpxGPU* Pi_Qr,
    cmpxGPU* Pi_Qrs,
    int Nx, int Ny, int Nz, int Nd,
    int Maxit, realGPU Tol,
    PROFILE* Profile);
#endif
