#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "FAME_Internal_Common.h"


void vec_plus(realCPU alpha, realCPU* vec1, realCPU beta, realCPU* vec2, int dim)
{
    for( int i = 0; i < dim; i++)
            vec1[i] = alpha * vec1[i] + beta * vec2[i];
}

void vec_plus(realCPU* vec_sum, realCPU alpha, realCPU* vec1, realCPU beta, realCPU* vec2, int dim)
{
    for( int i = 0; i < dim; i++)
            vec_sum[i] = alpha * vec1[i] + beta * vec2[i];
}

void vec_plus(realCPU* vec_sum, realCPU alpha, realCPU* vec1, realCPU beta, int* vec2, int dim)
{
    for( int i = 0; i < dim; i++)
            vec_sum[i] = alpha * vec1[i] + beta * (realCPU)vec2[i];
}

void vec_plus(realCPU* vec_sum, realCPU alpha, int* vec1, realCPU beta, int* vec2, int dim)
{
    for( int i = 0; i < dim; i++)
            vec_sum[i] = alpha * (realCPU)vec1[i] + beta * (realCPU)vec2[i];
}
void vec_plus(cmpxCPU* vec_sum, realCPU alpha, cmpxCPU* vec1, realCPU beta, cmpxCPU* vec2, int dim)
{
    for( int i = 0; i < dim; i++)
            vec_sum[i] = alpha * vec1[i] + beta * vec2[i];
}
void vec_plus(cmpxCPU* vec_sum, cmpxCPU alpha, cmpxCPU* vec1, cmpxCPU beta, cmpxCPU* vec2, int dim)
{
    for( int i = 0; i < dim; i++)
            vec_sum[i] = alpha * vec1[i] + beta * vec2[i];
             
}
