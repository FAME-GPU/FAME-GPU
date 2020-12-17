#ifndef _FAME_MATRIX_VECTOR_PRODUCTION_INVB_BIISOTROPIC_H_
#define _FAME_MATRIX_VECTOR_PRODUCTION_INVB_BIISOTROPIC_H_


int FAME_Matrix_Vector_Production_invB_Biisotropic( CULIB_HANDLES cuHandles,
                                                    MTX_B mtx_B,
                                                    int N,
                                                    cuDoubleComplex* vec_x,
                                                    cuDoubleComplex* vec_y);

#endif
