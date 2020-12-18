#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "FAME_Internal_Common.h"


// copy alpha * mtx_A into mtx_B
void mtx_cat(cmpxCPU alpha, cmpxCPU* mtx_A, cmpxCPU* mtx_B, int n, int m)
{
	for(int i = 0; i < n*m; i++)
		mtx_B[i] = alpha * mtx_A[i];
}

void mtx_cat(realCPU alpha, realCPU* mtx_A, realCPU* mtx_B, int n, int m)
{
	for(int i = 0; i < n * m; i++)
		mtx_B[i] = alpha * mtx_A[i];
}