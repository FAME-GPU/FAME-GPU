#ifndef _LANCZOS_DECOMP_GPU_H_
#define	_LANCZOS_DECOMP_GPU_H_

int Lanczos_decomp_gpu( 
    cuDoubleComplex* U,
    double*          Talpha,
    double*          Tbeta,
    bool             isInit,
    CULIB_HANDLES    cuHandles,
    FFT_BUFFER       fft_buffer,
    LAMBDAS_CUDA     Lambdas_cuda,
    MTX_B            mtx_B,
    LS               ls,
    int Nx, int Ny, int Nz, int Nd, int Nwant, int Nstep,
    string flag_CompType, PROFILE* Profile);

#endif