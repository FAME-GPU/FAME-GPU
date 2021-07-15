
#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include <complex.h>
#include "printDeviceArray.cuh"

void inv_3_Bianiso(cmpxCPU A[], cmpxCPU* result)
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


int FAME_Matrix_B_Bianisotropic(int N, MATERIAL material, cmpxGPU* G)
{
    
    cmpxCPU* varep_in = (cmpxCPU*) calloc(9, sizeof(cmpxCPU));
    cmpxCPU* xi_in = (cmpxCPU*) calloc(9, sizeof(cmpxCPU));
    cmpxCPU* zeta_in = (cmpxCPU*) calloc(9, sizeof(cmpxCPU));
    cmpxCPU* mu_in = (cmpxCPU*) calloc(9, sizeof(cmpxCPU));
    cmpxCPU* G_in = (cmpxCPU*) calloc(36, sizeof(cmpxCPU));

    for(int i = 0; i < 9; i++)
    {
        varep_in[i] = material.ele_permitt_in[i] + material.ele_permitt_in[i + 9] * I;
        xi_in[i] = material.reciprocity_in[i] + material.chirality_in[i] + material.chirality_in[i + 9] * I;
        zeta_in[i] = material.reciprocity_in[i] + material.chirality_in[i] - material.chirality_in[i + 9] * I;
        mu_in[i] = material.mag_permeab_in[i] + material.mag_permeab_in[i + 9] * I;
    }


	cmpxCPU rpig, rmig; 
    realCPU r2pg2;
	rmig = xi_in[0];
	rpig = zeta_in[0];
	r2pg2 = creal(rpig * rmig);
	cmpxCPU* tmp = (cmpxCPU*) calloc(9, sizeof(cmpxCPU));
	cmpxCPU* itmp = (cmpxCPU*) calloc(9, sizeof(cmpxCPU));
	
	tmp[0] = varep_in[0] - r2pg2;
	tmp[1] = varep_in[1];
	tmp[2] = varep_in[2];
	tmp[3] = varep_in[3];
	tmp[4] = varep_in[4] - r2pg2;
	tmp[5] = varep_in[5];
	tmp[6] = varep_in[6];
	tmp[7] = varep_in[7];
	tmp[8] = varep_in[8] - r2pg2;

	inv_3_Bianiso(tmp, itmp);

	for(int i = 0; i < 3; i++)
	{
		G_in[i] = itmp[i];
		G_in[i+6] = itmp[i+3];
		G_in[i+12] = itmp[i+6];
		
		G_in[i+3] = - rpig * itmp[i];
		G_in[i+9] = - rpig * itmp[i+3];
		G_in[i+15] = - rpig * itmp[i+6];

		G_in[i+18] = - rmig * itmp[i];
		G_in[i+24] = - rmig * itmp[i+3];
		G_in[i+30] = - rmig * itmp[i+6];

		G_in[i+21] = r2pg2 * itmp[i];
		G_in[i+27] = r2pg2 * itmp[i+3];
		G_in[i+33] = r2pg2 * itmp[i+6];
	}

	G_in[21] += 1;
	G_in[28] += 1;
	G_in[35] += 1;


    cudaMemcpy(G, G_in, 36*sizeof(cmpxGPU), cudaMemcpyHostToDevice);
    printDeviceArray(G, 36, "G.txt");

    return 0;

}