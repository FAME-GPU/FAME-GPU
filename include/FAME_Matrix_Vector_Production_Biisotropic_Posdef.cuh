#ifndef _FAME_MATRIX_VECTOR_PRODUCTION_BIISOTROPIC_POSDEF_H_
#define _FAME_MATRIX_VECTOR_PRODUCTION_BIISOTROPIC_POSDEF_H_
void FAME_Matrix_Vector_Production_Biisotropic_Posdef( CULIB_HANDLES cuHandles,  cmpxGPU* vec_x, MTX_B mtx_B, int Nx, int Ny,int Nz, int Nd, realGPU* Lambda_q_sqrt, cmpxGPU* Pi_Qr, cmpxGPU* Pi_Pr, cmpxGPU* Pi_Qrs, cmpxGPU* Pi_Prs, cmpxGPU* D_k, cmpxGPU* D_ks, cmpxGPU* vec_y);

#endif
