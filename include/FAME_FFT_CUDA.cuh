#ifndef _FAME_FFT_CUDA_H_
#define _FAME_FFT_CUDA_H_

int FFT_CUDA(cmpxGPU* vec_y, cmpxGPU* vec_x, cmpxGPU* dD_ks, FFT_BUFFER fftBuffer, CULIB_HANDLES cuHandles, int Nx, int Ny, int Nz);

int IFFT_CUDA(cmpxGPU* vec_y, cmpxGPU* vec_x, cmpxGPU* dD_k, FFT_BUFFER fftBuffer, CULIB_HANDLES cuHandles, int Nx, int Ny, int Nz);

int spMV_fastT_gpu( 
    cmpxGPU *out,
    cmpxGPU *p,
    CULIB_HANDLES cuHandles,
    FFT_BUFFER *fftBuffer,
    cmpxGPU *mtx_D_kx,   // D_kx
    cmpxGPU *mtx_D_jx,   // D_ky
    cmpxGPU *mtx_D_jell, // D_kz 
    const int n1,
    const int n2,
    const int n3,
    const int flag);
                        
#endif
