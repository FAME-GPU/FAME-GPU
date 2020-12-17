#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <string>
using namespace std;
typedef double _Complex cmpx;
// 2020-02-19

void mtx_prod(double* ans, double* M1, double* M2, int m, int n, int p)
{
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < p; j++)
        {
            ans[j * m + i] = 0.0;
            for(int k = 0; k < n; k++)
                ans[j * m + i] += M1[k * m + i] * M2[j * n + k];
        }
    }
}

void mtx_prod(cmpx* ans, cmpx* M1, double* M2, int m, int n, int p)
{
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < p; j++)
        {
            ans[j * m + i] = 0.0;
            for(int k = 0; k < n; k++)
                ans[j * m + i] += M1[k * m + i] * M2[j * n + k];
        }
    }
}

void mtx_prod(cmpx* ans, double* M1, cmpx* M2, int m, int n, int p)
{
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < p; j++)
        {
            ans[j * m + i] = 0.0;
            for(int k = 0; k < n; k++)
                ans[j * m + i] += M1[k * m + i] * M2[j * n + k];
        }
    }
}

void mtx_prod(cmpx* ans, cmpx* M1, cmpx* M2, int m, int n, int p)
{
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < p; j++)
        {
            ans[j * m + i] = 0.0;
            for(int k = 0; k < n; k++)
                ans[j * m + i] += M1[k * m + i] * M2[j * n + k];
        }
    }
}

void mtx_prod(cmpx* ans, int* M1_row, int* M1_col, cmpx* M1_val, cmpx* vec2, int nnz, int m)
{
    // Do the Matrix-vector production with sparse format matrix M1
    for( int i = 0; i < m; i++ )
        ans[i] = 0.0;
    
    for( int i = 0; i < nnz; i++ )
        ans[M1_row[i]] += M1_val[i]*vec2[M1_col[i]];
}

void mtx_prod(cmpx* ans, int* M1_row, int* M1_col, cmpx* M1_val, cmpx* vec2, int nnz, int m, string flag_CompType)
{
    // Do the Matrix-vector production with sparse format matrix M1
	if (flag_CompType == "Conjugate Transpose")
	{
    	for( int i = 0; i < m; i++ )
        	ans[i] = 0.0;
    
    	for( int i = 0; i < nnz; i++ )
        	ans[M1_col[i]] += conj(M1_val[i])*vec2[M1_row[i]];
	}
	else
		printf("mtx_prod flag_CompType error!\n");
}
