#ifndef _FAME_MATRIX_VECTOR_PRODUCTION_BIISOTROPIC_POSDEF_H_
#define _FAME_MATRIX_VECTOR_PRODUCTION_BIISOTROPIC_POSDEF_H_
void FAME_Matrix_Vector_Production_Biisotropic_Posdef( CULIB_HANDLES cuHandles,  cuDoubleComplex* vec_x, MTX_B mtx_B, int Nx, int Ny,int Nz, int Nd, double* Lambda_q_sqrt, cuDoubleComplex* Pi_Qr, cuDoubleComplex* Pi_Pr, cuDoubleComplex* Pi_Qrs, cuDoubleComplex* Pi_Prs, cuDoubleComplex* D_k, cuDoubleComplex* D_ks, cuDoubleComplex* vec_y);

#endif
