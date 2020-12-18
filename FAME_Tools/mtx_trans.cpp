#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "FAME_Internal_Common.h"


// return the transpose of nxm matrix mtx_A into mtx_B
void mtx_trans(cmpxCPU* mtx_A, cmpxCPU* mtx_B, int n, int m)
{
	for( int j = 0; j < n; j++)
		for( int i = 0; i < m; i++)
			mtx_B[i+j*m] = mtx_A[j+i*n];
}

void mtx_trans(realCPU* mtx_A, realCPU* mtx_B, int n, int m)
{
	for( int j = 0; j < n; j++)
		for( int i = 0; i < m; i++)
			mtx_B[i+j*m] = mtx_A[j+i*n];
}
