#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
typedef double _Complex cmpx;
// 2020-02-19

void mtx_print(double* M, int m, int n)
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

void mtx_print(cmpx* M, int m, int n)
{
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
            printf("( % e  + 1i*(% e) )\t", creal(M[j * m + i]), cimag(M[j * m + i]));
        printf("\n");
    }
}
