#ifndef _FAME_MATRIX_VECTOR_PRODUCTION_BIISOTROPIC_AR_
#define _FAME_MATRIX_VECTOR_PRODUCTION_BIISOTROPIC_AR_
void FAME_Matrix_Vector_Production_Biisotropic_Ar(CULIB_HANDLES cuHandles, FFT_BUFFER fft_buffer, 
     cuDoubleComplex* vec_x, MTX_B mtx_B, int Nx, int Ny, int Nz, int Nd,
      cuDoubleComplex* Pi_Qr, cuDoubleComplex* Pi_Pr, cuDoubleComplex* Pi_Qrs, cuDoubleComplex* Pi_Prs,
       cuDoubleComplex* D_k, cuDoubleComplex* D_ks, cuDoubleComplex* vec_y, PROFILE* Profile);
void FAME_Matrix_Vector_Production_Biisotropic_Ar(CULIB_HANDLES cuHandles, FFT_BUFFER fft_buffer,
    cuDoubleComplex* vec_x, MTX_B mtx_B, int Nx, int Ny, int Nz, int Nd, 
    cuDoubleComplex* Pi_Qr, cuDoubleComplex* Pi_Pr,cuDoubleComplex* Pi_Qrs, cuDoubleComplex* Pi_Prs, 
    cuDoubleComplex* D_kx, cuDoubleComplex* D_ky, cuDoubleComplex* D_kz, cuDoubleComplex* vec_y, PROFILE* Profile);
#endif
