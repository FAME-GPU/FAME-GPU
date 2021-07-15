#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"

int FAME_Matrix_B_Isotropic(realCPU* dB_eps, realCPU* dinvB_eps, MATERIAL material, int N)
{
	int i, j;
	int N3 = N * 3;

	realCPU* B_eps    = (realCPU*) malloc(N3 * sizeof(realCPU));
	realCPU* invB_eps = (realCPU*) malloc(N3 * sizeof(realCPU));

	realCPU temp_ele_permitt_in = material.ele_permitt_in[0];

	int temp = N * material.material_num;

	for(i = 0; i < N; i++)
    {
    	   B_eps[i      ] =       material.ele_permitt_out;
           B_eps[i +   N] =       material.ele_permitt_out;
           B_eps[i + 2*N] =       material.ele_permitt_out;
        invB_eps[i      ] = 1.0 / material.ele_permitt_out;
        invB_eps[i +   N] = 1.0 / material.ele_permitt_out;
        invB_eps[i + 2*N] = 1.0 / material.ele_permitt_out;

        for(j = 0; j < material.material_num; j++)
        {
            if(material.Binout[i + j*N] == 1)
            {
            	   B_eps[i] =     temp_ele_permitt_in;
                invB_eps[i] = 1.0/temp_ele_permitt_in;
            }
            
            if(material.Binout[i + j*N +   temp] == 1)
            {
            	   B_eps[i + N] =     temp_ele_permitt_in;
                invB_eps[i + N] = 1.0/temp_ele_permitt_in;
            }
            
            if(material.Binout[i + j*N + 2*temp] == 1)
            {
            	   B_eps[i + 2*N] =     temp_ele_permitt_in;
                invB_eps[i + 2*N] = 1.0/temp_ele_permitt_in;
            }
        }
    }

	checkCudaErrors(cudaMemcpy(dB_eps,       B_eps, N3 * sizeof(realCPU), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dinvB_eps, invB_eps, N3 * sizeof(realCPU), cudaMemcpyHostToDevice));

	free(B_eps);
	free(invB_eps);
	free(material.Binout);
	free(material.ele_permitt_in);

	return 0;
}
