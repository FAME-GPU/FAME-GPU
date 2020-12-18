#include "FAME_Internal_Common.h"
#include <complex.h>

int FAME_Create_C_txt(
    int* C1_r, int* C1_c, cmpxCPU* C1_v,
    int* C2_r, int* C2_c, cmpxCPU* C2_v,
    int* C3_r, int* C3_c, cmpxCPU* C3_v,
    int* grid_nums)
{
    int i;
    int N = grid_nums[0] * grid_nums[1] * grid_nums[2];

    FILE* fp;
    fp = fopen("Data_Mtx_C.txt", "w");

    fprintf(fp, "#[ C1_row, C1_col, C1_val ] = \n");
    for (i = 0; i < 2*N; i++)
      //  fprintf(fp, "%d %d %f+1i*(%f)\n", C1_r[i], C1_c[i], creal(C1_v[i]), cimag(C1_v[i]) );
        fprintf(fp, "%d %d %f\n", C1_r[i]+1, C1_c[i]+1, creal(C1_v[i]));
    fprintf(fp, "\n\n");

    fprintf(fp, "#[ C2_row, C2_col, C2_val ] = \n");
    for (i = 0; i < 2*N; i++)
     //   fprintf(fp, "%d %d %f+1i*(%f)\n", C2_r[i], C2_c[i], creal(C2_v[i]), cimag(C2_v[i]) );
        fprintf(fp, "%d %d %f\n", C2_r[i]+1, C2_c[i]+1, creal(C2_v[i]));
    fprintf(fp, "\n\n");

    fprintf(fp, "#[ C3_row, C3_col, C3_val ] = \n");
    for (i = 0; i < 2*N; i++)
       // fprintf(fp, "%d %d %f+1i*(%f)\n", C3_r[i], C3_c[i], creal(C3_v[i]), cimag(C3_v[i]) );
        fprintf(fp, "%d %d %f\n", C3_r[i]+1, C3_c[i]+1, creal(C3_v[i]));

    fclose(fp);

    return 0;
}
