#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
typedef double _Complex cmpx;
// 2020-02-19

// return the transpose of nxm matrix mtx_A into mtx_B
void mtx_trans_conj(cmpx* mtx_A, cmpx* mtx_B, int n, int m)
{
	for( int j = 0; j < n; j++)
		for( int i = 0; i < m; i++)
			mtx_B[i+j*m] = conj(mtx_A[j+i*n]);
}

// return the transpose of nxm sparse matrix mtx_A into mtx_B
void mtx_trans_conj(int* mtx_A_r, int* mtx_A_c, cmpx* mtx_A_v, 
	                int* mtx_B_r, int* mtx_B_c, cmpx* mtx_B_v, 
	                int  nnz)
{
	for(int i = 0; i < nnz; i++)
	{
		mtx_B_r[i] = mtx_A_c[i];
		mtx_B_c[i] = mtx_A_r[i];
		mtx_B_v[i] = conj(mtx_A_v[i]);
	}
}
