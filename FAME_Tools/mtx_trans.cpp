#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
typedef double _Complex cmpx;
// 2020-02-19

// return the transpose of nxm matrix mtx_A into mtx_B
void mtx_trans(cmpx* mtx_A, cmpx* mtx_B, int n, int m)
{
	for( int j = 0; j < n; j++)
		for( int i = 0; i < m; i++)
			mtx_B[i+j*m] = mtx_A[j+i*n];
}

void mtx_trans(double* mtx_A, double* mtx_B, int n, int m)
{
	for( int j = 0; j < n; j++)
		for( int i = 0; i < m; i++)
			mtx_B[i+j*m] = mtx_A[j+i*n];
}
