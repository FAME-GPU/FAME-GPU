#ifndef _FAME_MATRIX_VECTOR_PRODUCTION_BIANISOTROPIC_DUALG_SHUFFLE_H_
#define _FAME_MATRIX_VECTOR_PRODUCTION_BIANISOTROPIC_DUALG_SHUFFLE_H_

void FAME_Matrix_Vector_Production_Bianisotropic_dualG_Shuffle(CULIB_HANDLES cuHandles, int Nx, int Ny, int Nz, int Nd, 
    int* InOut_index, int* InOut_index_length, cmpxGPU* G_in, cmpxGPU* vec);

#endif