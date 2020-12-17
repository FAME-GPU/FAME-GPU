#ifndef _FAME_DESTROY_CUDA_H_
#define _FAME_DESTROY_CUDA_H_

int FAME_Destroy_Main(
    CULIB_HANDLES  cuHandles,
    FFT_BUFFER     fft_buffer,
    LANCZOS_BUFFER lBuffer,
    MTX_B          mtx_B,
    MTX_C          mtx_C,
    double*        Freq_array,
    cmpx*          Ele_field_mtx);
    
#endif
