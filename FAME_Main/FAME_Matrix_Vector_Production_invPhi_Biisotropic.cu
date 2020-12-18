#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include <complex.h>


static __global__ void dot_pro( int size,
                                realGPU* B_invPhi,
                                cmpxGPU* temp_vec_y_ele,
                                cmpxGPU* vec_y_ele);

void FAME_Matrix_Vector_Production_invPhi_Biisotropic
	(CULIB_HANDLES cuHandles, int size, realGPU* invPhi, cmpxGPU* vec_y_ele_in, cmpxGPU* vec_y_ele_out)
{

	dim3 DimBlock(BLOCK_SIZE,1,1);
    dim3 DimGrid((size-1)/BLOCK_SIZE +1,1,1);

	dot_pro<<<DimGrid, DimBlock>>>( size, invPhi, vec_y_ele_in, vec_y_ele_out);

}

static __global__ void dot_pro( int size,
                                realGPU* B_invPhi,
                                cmpxGPU* temp_vec_y_ele,
                                cmpxGPU* vec_y_ele)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if( idx < size )
    {
        vec_y_ele[idx].x = B_invPhi[idx]*temp_vec_y_ele[idx].x;
        vec_y_ele[idx].y = B_invPhi[idx]*temp_vec_y_ele[idx].y;
    }

}

