#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "FAME_Internal_Common.h"


// Compute the dot product of mtx_A and mtx_B with dimension n*m and point to mtx_B
void mtx_dot_prod(cmpxCPU* mtx_A, cmpxCPU* mtx_B, int n, int m)
{
	for( int i = 0; i < n*m; i++)
		mtx_B[i] = mtx_A[i]*mtx_B[i];
}

void mtx_dot_prod(realCPU* mtx_A, realCPU* mtx_B, int n, int m)
{
	for( int i = 0; i < n*m; i++)
		mtx_B[i] = mtx_A[i]*mtx_B[i];
}

void mtx_dot_prod(realCPU* mtx_A, cmpxCPU* mtx_B, int n, int m)
{
	for( int i = 0; i < n*m; i++)
		mtx_B[i] = mtx_A[i]*mtx_B[i];
}

void mtx_dot_prod(cmpxCPU* ans, realCPU* mtx_A, cmpxCPU* mtx_B, int n, int m)
{
	for( int i = 0; i < n*m; i++)
		ans[i] = mtx_A[i]*mtx_B[i];
}

void mtx_dot_prod(realCPU* mtx_A, cmpxCPU* mtx_B, cmpxCPU* result, int n, int m)
{
	for( int i = 0; i < n*m; i++)
		result[i] = mtx_A[i]*mtx_B[i];
}

void mtx_dot_prod(cmpxCPU* mtx_A, cmpxCPU* mtx_B, cmpxCPU* result, int n, int m)
{
	for( int i = 0; i < n*m; i++)
		result[i] = mtx_A[i]*mtx_B[i];
}