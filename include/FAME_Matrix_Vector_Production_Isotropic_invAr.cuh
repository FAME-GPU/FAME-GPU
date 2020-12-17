#ifndef _FAME_MATRIX_VECTOR_PRODUCTION_ISOTROPIC_INVAR_H_
#define _FAME_MATRIX_VECTOR_PRODUCTION_ISOTROPIC_INVAR_H_

int FAME_Matrix_Vector_Production_Isotropic_invAr(
      cuDoubleComplex* vec_y,
      cuDoubleComplex* vec_x,
      CULIB_HANDLES    cuHandles,
      FFT_BUFFER       fft_buffer,
      LAMBDAS_CUDA     Lambdas_cuda,
      MTX_B            mtx_B,
      LS               ls,
      int Nx, int Ny, int Nz, int Nd,
      string flag_CompType, PROFILE* Profile);

#endif
