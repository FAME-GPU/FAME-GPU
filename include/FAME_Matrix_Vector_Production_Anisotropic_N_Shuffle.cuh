#ifndef _FAME_MATRIX_VECTOR_PRODUCTION_ANISOTROPIC_N_SHUFFLE_H_
#define _FAME_MATRIX_VECTOR_PRODUCTION_ANISOTROPIC_N_SHUFFLE_H_

int FAME_Matrix_Vector_Production_Anisotropic_N_Shuffle(CULIB_HANDLES cuHandles, int Nx, int Ny, int Nz, int Nd, 
    int *InOut_index, int *InOut_index_length, cmpxGPU* N_in, cmpxGPU* vec);

#endif