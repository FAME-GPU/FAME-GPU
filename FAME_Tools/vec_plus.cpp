#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
typedef double _Complex cmpx;
// 2020-02-19

void vec_plus(double* vec_sum, double alpha, double* vec1, double beta, double* vec2, int dim)
{
    for( int i = 0; i < dim; i++)
            vec_sum[i] = alpha * vec1[i] + beta * vec2[i];
}

void vec_plus(double* vec_sum, double alpha, double* vec1, double beta, int* vec2, int dim)
{
    for( int i = 0; i < dim; i++)
            vec_sum[i] = alpha * vec1[i] + beta * (double)vec2[i];
}

void vec_plus(double* vec_sum, double alpha, int* vec1, double beta, int* vec2, int dim)
{
    for( int i = 0; i < dim; i++)
            vec_sum[i] = alpha * (double)vec1[i] + beta * (double)vec2[i];
}
void vec_plus(cmpx* vec_sum, double alpha, cmpx* vec1, double beta, cmpx* vec2, int dim)
{
    for( int i = 0; i < dim; i++)
            vec_sum[i] = alpha * vec1[i] + beta * vec2[i];
}
void vec_plus(cmpx* vec_sum, cmpx alpha, cmpx* vec1, cmpx beta, cmpx* vec2, int dim)
{
    for( int i = 0; i < dim; i++)
            vec_sum[i] = alpha * vec1[i] + beta * vec2[i];
             
}