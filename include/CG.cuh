#ifndef _CG_H_
#define _CG_H_
int CG(
    cuDoubleComplex* vec_y,
    cuDoubleComplex* b,
    CULIB_HANDLES    cuHandles,
    FFT_BUFFER       fft_buffer,
    MTX_B            mtx_B,
    cuDoubleComplex* D_k,
    cuDoubleComplex* D_ks,
    cuDoubleComplex* Pi_Qr,
    cuDoubleComplex* Pi_Qrs,
    int Nx, int Ny, int Nz, int Nd,
    int Maxit, double Tol,
    PROFILE* Profile);

int CG(
    cuDoubleComplex* vec_y,
    cuDoubleComplex* b,
    CULIB_HANDLES    cuHandles,
    FFT_BUFFER       fft_buffer,
    MTX_B            mtx_B,
    cuDoubleComplex* D_kx,
    cuDoubleComplex* D_ky,
    cuDoubleComplex* D_kz,
    cuDoubleComplex* Pi_Qr,
    cuDoubleComplex* Pi_Qrs,
    int Nx, int Ny, int Nz, int Nd,
    int Maxit, double Tol,
    PROFILE* Profile);
#endif
