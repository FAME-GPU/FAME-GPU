#ifndef _FAME_MATRIX_VECTOR_PRODUCTION_ISOTROPIC_QBQ_H_
#define _FAME_MATRIX_VECTOR_PRODUCTION_ISOTROPIC_QBQ_H_

int FAME_Matrix_Vector_Production_Isotropic_QBQ(
      cuDoubleComplex* vec_y,
      cuDoubleComplex* vec_x,
      CULIB_HANDLES    cuHandles,
      FFT_BUFFER       fft_buffer,
      MTX_B            mtx_B,
      cuDoubleComplex* D_k,
      cuDoubleComplex* D_ks,
      cuDoubleComplex* Pi_Qr,
      cuDoubleComplex* Pi_Qrs,
      int Nx, int Ny, int Nz, int Nd,
      PROFILE* Profile);

int FAME_Matrix_Vector_Production_Isotropic_QBQ(
      cuDoubleComplex* vec_y,
      cuDoubleComplex* vec_x,
      CULIB_HANDLES    cuHandles,
      FFT_BUFFER       fft_buffer,
      MTX_B            mtx_B,
      cuDoubleComplex* D_kx,
      cuDoubleComplex* D_ky,
      cuDoubleComplex* D_kz,
      cuDoubleComplex* Pi_Qr,
      cuDoubleComplex* Pi_Qrs,
      int Nx, int Ny, int Nz, int Nd,
      PROFILE* Profile);
      
#endif
