#ifndef _FAME_FFT_CUDA_H_
#define _FAME_FFT_CUDA_H_

int FFT_CUDA(cuDoubleComplex* vec_y, cuDoubleComplex* vec_x, cuDoubleComplex* dD_ks, FFT_BUFFER fftBuffer, CULIB_HANDLES cuHandles, int Nx, int Ny, int Nz);

int IFFT_CUDA(cuDoubleComplex* vec_y, cuDoubleComplex* vec_x, cuDoubleComplex* dD_k, FFT_BUFFER fftBuffer, CULIB_HANDLES cuHandles, int Nx, int Ny, int Nz);

int spMV_fastT_gpu( 
    cuDoubleComplex *out,
    cuDoubleComplex *p,
    CULIB_HANDLES cuHandles,
    FFT_BUFFER *fftBuffer,
    cuDoubleComplex *mtx_D_kx,   // D_kx
    cuDoubleComplex *mtx_D_jx,   // D_ky
    cuDoubleComplex *mtx_D_jell, // D_kz 
    const int n1,
    const int n2,
    const int n3,
    const int flag);
                        
#endif
