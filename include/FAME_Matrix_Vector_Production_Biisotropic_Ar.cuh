#ifndef _FAME_MATRIX_VECTOR_PRODUCTION_BIISOTROPIC_AR_
#define _FAME_MATRIX_VECTOR_PRODUCTION_BIISOTROPIC_AR_
void FAME_Matrix_Vector_Production_Biisotropic_Ar(CULIB_HANDLES cuHandles, FFT_BUFFER fft_buffer, 
     cmpxGPU* vec_x, MTX_B mtx_B, int Nx, int Ny, int Nz, int Nd,
      cmpxGPU* Pi_Qr, cmpxGPU* Pi_Pr, cmpxGPU* Pi_Qrs, cmpxGPU* Pi_Prs,
       cmpxGPU* D_k, cmpxGPU* D_ks, cmpxGPU* vec_y, PROFILE* Profile);
void FAME_Matrix_Vector_Production_Biisotropic_Ar(CULIB_HANDLES cuHandles, FFT_BUFFER fft_buffer,
    cmpxGPU* vec_x, MTX_B mtx_B, int Nx, int Ny, int Nz, int Nd, 
    cmpxGPU* Pi_Qr, cmpxGPU* Pi_Pr,cmpxGPU* Pi_Qrs, cmpxGPU* Pi_Prs, 
    cmpxGPU* D_kx, cmpxGPU* D_ky, cmpxGPU* D_kz, cmpxGPU* vec_y, PROFILE* Profile);
#endif
