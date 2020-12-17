#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
typedef double _Complex cmpx;
// 2020-02-19

// copy alpha * mtx_A into mtx_B
void mtx_cat(cmpx alpha, cmpx* mtx_A, cmpx* mtx_B, int n, int m)
{
	for(int i = 0; i < n*m; i++)
		mtx_B[i] = alpha * mtx_A[i];
}

void mtx_cat(double alpha, double* mtx_A, double* mtx_B, int n, int m)
{
	for(int i = 0; i < n * m; i++)
		mtx_B[i] = alpha * mtx_A[i];
}