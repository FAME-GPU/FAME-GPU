#ifndef _FAME_MATRIX_VECTOR_PRODUCTION_ANISOTROPIC_QBQ_H_
#define _FAME_MATRIX_VECTOR_PRODUCTION_ANISOTROPIC_QBQ_H_

int FAME_Matrix_Vector_Production_Anisotropic_QBQ(
      cmpxGPU* vec_y,
      cmpxGPU* vec_x,
      CULIB_HANDLES    cuHandles,
      FFT_BUFFER       fft_buffer,
      MTX_B            mtx_B,
      cmpxGPU* D_k,
      cmpxGPU* D_ks,
      cmpxGPU* Pi_Qr,
      cmpxGPU* Pi_Qrs,
      cmpxGPU* Pi_Qr_110,
      cmpxGPU* Pi_Qrs_110,
      cmpxGPU* Pi_Qr_101,
      cmpxGPU* Pi_Qrs_101,
      cmpxGPU* Pi_Qr_011,
      cmpxGPU* Pi_Qrs_011,
      int Nx, int Ny, int Nz, int Nd);

int FAME_Matrix_Vector_Production_Anisotropic_QBQ(
      cmpxGPU* vec_y,
      cmpxGPU* vec_x,
      CULIB_HANDLES    cuHandles,
      FFT_BUFFER       fft_buffer,
      MTX_B            mtx_B,
      cmpxGPU* D_kx,
      cmpxGPU* D_ky,
      cmpxGPU* D_kz,
      cmpxGPU* Pi_Qr,
      cmpxGPU* Pi_Qrs,
      cmpxGPU* Pi_Qr_110,
      cmpxGPU* Pi_Qrs_110,
      cmpxGPU* Pi_Qr_101,
      cmpxGPU* Pi_Qrs_101,
      cmpxGPU* Pi_Qr_011,
      cmpxGPU* Pi_Qrs_011,
      int Nx, int Ny, int Nz, int Nd);
      
#endif
