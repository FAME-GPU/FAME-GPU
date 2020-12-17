#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"

int FAME_Matrix_B_Isotropic(double* dB_eps, double* dinvB_eps, MATERIAL material, int N)
{
	int i, j;
	int N3 = N * 3;

	double* B_eps    = (double*) malloc(N3 * sizeof(double));
	double* invB_eps = (double*) malloc(N3 * sizeof(double));
	double* temp_ele_permitt_in;

	if (material.material_num > material.num_ele_permitt_in)
	{
		temp_ele_permitt_in = (double*) malloc(material.material_num * sizeof(double));
		printf("The input number of num_ele_permitt_in (%d) is less than material_num (%d).\n", material.num_ele_permitt_in, material.material_num);
		
		printf("Please input %d ele_permitt_in (in double precision).\n", material.material_num);
		for(i = 0; i <material.material_num; i++)
		{
			printf("No.%d ele_permitt_in : ", i + 1);
			scanf("%lf", &temp_ele_permitt_in[i]);
		}
	}
	else
	{
		temp_ele_permitt_in = (double*) malloc(material.num_ele_permitt_in * sizeof(double));
		memcpy(temp_ele_permitt_in, material.ele_permitt_in, material.num_ele_permitt_in * sizeof(double));
	}

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
            	   B_eps[i] =     temp_ele_permitt_in[j];
                invB_eps[i] = 1.0/temp_ele_permitt_in[j];
            }
            
            if(material.Binout[i + j*N +   temp] == 1)
            {
            	   B_eps[i + N] =     temp_ele_permitt_in[j];
                invB_eps[i + N] = 1.0/temp_ele_permitt_in[j];
            }
            
            if(material.Binout[i + j*N + 2*temp] == 1)
            {
            	   B_eps[i + 2*N] =     temp_ele_permitt_in[j];
                invB_eps[i + 2*N] = 1.0/temp_ele_permitt_in[j];
            }
        }
    }

	checkCudaErrors(cudaMemcpy(dB_eps,       B_eps, N3 * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dinvB_eps, invB_eps, N3 * sizeof(double), cudaMemcpyHostToDevice));

	free(B_eps);
	free(invB_eps);
	free(material.Binout);
	free(material.ele_permitt_in);
	free(temp_ele_permitt_in);

	return 0;
}
