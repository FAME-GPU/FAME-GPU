#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
typedef double _Complex cmpx;
// 2020-02-19

// Compute the dot product of mtx_A and mtx_B with dimension n*m and point to mtx_B
void mtx_dot_prod(cmpx* mtx_A, cmpx* mtx_B, int n, int m)
{
	for( int i = 0; i < n*m; i++)
		mtx_B[i] = mtx_A[i]*mtx_B[i];
}

void mtx_dot_prod(double* mtx_A, double* mtx_B, int n, int m)
{
	for( int i = 0; i < n*m; i++)
		mtx_B[i] = mtx_A[i]*mtx_B[i];
}

void mtx_dot_prod(double* mtx_A, cmpx* mtx_B, int n, int m)
{
	for( int i = 0; i < n*m; i++)
		mtx_B[i] = mtx_A[i]*mtx_B[i];
}

void mtx_dot_prod(cmpx* ans, double* mtx_A, cmpx* mtx_B, int n, int m)
{
	for( int i = 0; i < n*m; i++)
		ans[i] = mtx_A[i]*mtx_B[i];
}

void mtx_dot_prod(double* mtx_A, cmpx* mtx_B, cmpx* result, int n, int m)
{
	for( int i = 0; i < n*m; i++)
		result[i] = mtx_A[i]*mtx_B[i];
}

void mtx_dot_prod(cmpx* mtx_A, cmpx* mtx_B, cmpx* result, int n, int m)
{
	for( int i = 0; i < n*m; i++)
		result[i] = mtx_A[i]*mtx_B[i];
}