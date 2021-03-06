
#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include <complex.h>
#include "printDeviceArray.cuh"

void inv_3_Aniso(cmpxCPU A[], cmpxCPU* result)
{
    cmpxCPU det = A[0] * (A[4] * A[8] - A[5] * A[7]) - A[3] * (A[1] * A[8] - A[7] * A[2]) + A[6] * (A[1] * A[5] - A[4] * A[2]);
    cmpxCPU invdet = 1 / det;
    result[0] =  (A[4] * A[8] - A[5] * A[7]) * invdet;
    result[3] = -(A[3] * A[8] - A[6] * A[5]) * invdet;
    result[6] =  (A[3] * A[7] - A[6] * A[4]) * invdet;
    result[1] = -(A[1] * A[8] - A[2] * A[7]) * invdet;
    result[4] =  (A[0] * A[8] - A[6] * A[2]) * invdet;
    result[7] = -(A[0] * A[7] - A[1] * A[6]) * invdet;
    result[2] =  (A[1] * A[5] - A[2] * A[4]) * invdet;
    result[5] = -(A[0] * A[5] - A[2] * A[3]) * invdet;
    result[8] =  (A[0] * A[4] - A[1] * A[3]) * invdet;
}


int FAME_Matrix_B_Anisotropic(int N, MATERIAL material, cmpxGPU* NN)
{
    
    cmpxCPU* varep_in = (cmpxCPU*) calloc(9, sizeof(cmpxCPU));
    cmpxCPU* N_in = (cmpxCPU*) calloc(9, sizeof(cmpxCPU));

    for(int i = 0; i < 9; i++)
    {
        varep_in[i] = material.ele_permitt_in[i] + material.ele_permitt_in[i + 9] * I;
    }
    
	inv_3_Aniso(varep_in, N_in);

    cudaMemcpy(NN, N_in, 9*sizeof(cmpxGPU), cudaMemcpyHostToDevice);
    // printDeviceArray(NN, 9, "N.txt");

    free(varep_in);
    free(N_in);

    return 0;

}