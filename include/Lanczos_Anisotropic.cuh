#ifndef _LANCZOS_ANISOTROPIC_H_
#define _LANCZOS_ANISOTROPIC_H_

int Lanczos_Anisotropic( 
    realCPU*          Freq_array, 
    cmpxGPU* ev,
    CULIB_HANDLES    cuHandles,
    LANCZOS_BUFFER   lBuffer,
    FFT_BUFFER       fft_buffer,
    LAMBDAS_CUDA     Lambdas_cuda,
    MTX_B            mtx_B,
    ES               es,
    LS               ls,
    int Nx, int Ny, int Nz, int Nd,
    string flag_CompType, PROFILE* Profile);

#endif
