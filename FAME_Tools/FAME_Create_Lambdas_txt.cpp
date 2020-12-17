#include "FAME_Internal_Common.h"
#include <complex.h>

int FAME_Create_Lambdas_txt(LAMBDAS Lambdas, int N)
{
    int i;
    
    FILE* fp;
    fp = fopen("Data_Mtx_Lambdas.txt", "w");

    fprintf(fp, "# Lambda_x = \n");
    for (i = 0; i < N; i++)
        fprintf(fp, "%f  %f\n", creal(Lambdas.Lambda_x[i]), cimag(Lambdas.Lambda_x[i]) );
    fprintf(fp, "\n\n");

    fprintf(fp, "# Lambda_y = \n");
    for (i = 0; i < N; i++)
        fprintf(fp, "%f  %f\n", creal(Lambdas.Lambda_y[i]), cimag(Lambdas.Lambda_y[i]) );
    fprintf(fp, "\n\n");

    fprintf(fp, "# Lambda_z = \n");
    for (i = 0; i < N; i++)
        fprintf(fp, "%f  %f \n", creal(Lambdas.Lambda_z[i]), cimag(Lambdas.Lambda_z[i]) );
    fprintf(fp, "\n\n");

    fclose(fp);

    return 0;
}
