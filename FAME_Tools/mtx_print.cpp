#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "FAME_Internal_Common.h"

void mtx_print(realCPU* M, int m, int n)
{
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
            printf("% lf ", M[j * m + i]);
        printf("\n");
    }
}

void mtx_print(int* M, int m, int n)
{
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
            printf("% ld ", M[j * m + i]);
        printf("\n");
    }
}

void mtx_print(cmpxCPU* M, int m, int n)
{
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
            printf("( % e  + 1i*(% e) )\t", creal(M[j * m + i]), cimag(M[j * m + i]));
        printf("\n");
    }
}
