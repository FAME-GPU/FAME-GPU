#ifndef _FAME_MATRIX_VECTOR_PRODUCTION_ISOTROPIC_QBQ_H_
#define _FAME_MATRIX_VECTOR_PRODUCTION_ISOTROPIC_QBQ_H_

int FAME_Matrix_Vector_Production_Isotropic_QBQ(
      cmpxGPU* vec_y,
      cmpxGPU* vec_x,
      CULIB_HANDLES    cuHandles,
      FFT_BUFFER       fft_buffer,
      MTX_B            mtx_B,
      cmpxGPU* D_k,
      cmpxGPU* D_ks,
      cmpxGPU* Pi_Qr,
      cmpxGPU* Pi_Qrs,
      int Nx, int Ny, int Nz, int Nd,
      PROFILE* Profile);

int FAME_Matrix_Vector_Production_Isotropic_QBQ(
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
      int Nx, int Ny, int Nz, int Nd,
      PROFILE* Profile);
      
#endif
