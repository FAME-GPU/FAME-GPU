#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <string>
using namespace std;
#include "FAME_Internal_Common.h"
// 2020-02-19

void mtx_prod(realCPU* ans, realCPU* M1, realCPU* M2, int m, int n, int p)
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

void mtx_prod(cmpxCPU* ans, cmpxCPU* M1, realCPU* M2, int m, int n, int p)
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

void mtx_prod(cmpxCPU* ans, realCPU* M1, cmpxCPU* M2, int m, int n, int p)
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

void mtx_prod(cmpxCPU* ans, cmpxCPU* M1, cmpxCPU* M2, int m, int n, int p)
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

void mtx_prod(cmpxCPU* ans, int* M1_row, int* M1_col, cmpxCPU* M1_val, cmpxCPU* vec2, int nnz, int m)
{
    // Do the Matrix-vector production with sparse format matrix M1
    for( int i = 0; i < m; i++ )
        ans[i] = 0.0;
    
    for( int i = 0; i < nnz; i++ )
        ans[M1_row[i]] += M1_val[i]*vec2[M1_col[i]];
}

void mtx_prod(cmpxCPU* ans, int* M1_row, int* M1_col, cmpxCPU* M1_val, cmpxCPU* vec2, int nnz, int m, string flag_CompType)
{
    // Do the Matrix-vector production with sparse format matrix M1
	if (flag_CompType == "Conjugate Transpose")
	{
    	for( int i = 0; i < m; i++ )
        	ans[i] = 0.0;
    
    	for( int i = 0; i < nnz; i++ )
        	ans[M1_col[i]] += conj(M1_val[i])*vec2[M1_row[i]];
	}
	else if(flag_CompType == "minus")
    {
        for( int i = 0; i < m; i++ )
            ans[i] = 0.0;
    
        for( int i = 0; i < nnz; i++ )
            ans[M1_row[i]] -= M1_val[i]*vec2[M1_col[i]];
    }
    else
		printf("mtx_prod flag_CompType error!\n");
}

void mtx_prod(cmpxCPU* ans, int* M1_row, int* M1_col, cmpxCPU* M1_val, cmpxCPU* vec2, int nnz, int m, string flag_CompType1, string flag_CompType2)
{
    // Do the Matrix-vector production with sparse format matrix M1
	if (flag_CompType1 == "minus" && flag_CompType2 == "Conjugate Transpose")
	{
    	for( int i = 0; i < m; i++ )
        	ans[i] = 0.0;
    
    	for( int i = 0; i < nnz; i++ )
        	ans[M1_col[i]] -= conj(M1_val[i])*vec2[M1_row[i]];
	}
	else
		printf("mtx_prod flag_CompType error!\n");
}
