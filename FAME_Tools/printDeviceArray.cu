#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <assert.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
typedef double _Complex cmpx;
// 2020-02-19

void printDeviceArray(double *d_Array, int n, const char *filename)
{
	cudaError_t cudaErr;

    FILE *fp = fopen(filename, "w");
    assert( fp != NULL );

    printf("Write array into %s.\n", filename);

    double *h_Array = (double*) malloc( n * sizeof(double) );
    cudaErr = cudaMemcpy(h_Array, d_Array, n*sizeof(double), cudaMemcpyDeviceToHost);
	assert( cudaErr == cudaSuccess );

	for(int i = 0 ; i < n; i++)
        fprintf(fp, "%15.18lf\n", h_Array[i]);
        
    fclose(fp);
    free(h_Array);
}

void printDeviceArray(cuDoubleComplex *d_Array, int n, const char *filename)
{
	cudaError_t cudaErr;
    FILE *fp = fopen(filename, "w");
    assert( fp != NULL );

    printf("Write array into %s.\n", filename);

    cmpx *h_Array = (cmpx*) malloc( n * sizeof(cmpx) );
    
    cudaErr = cudaMemcpy(h_Array, d_Array, n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    //printf("cudaErr %s.\n" , cudaGetErrorString(cudaErr));
	assert( cudaErr == cudaSuccess );

    for(int i = 0; i < n; i++)
       fprintf(fp, "%15.18lf\t %15.18lf\n", creal(h_Array[i]), cimag(h_Array[i]) );

    fclose(fp);
    free(h_Array);
}

void printDeviceArray(int *d_Array, int n, const char *filename)
{
	cudaError_t cudaErr;
    FILE *fp = fopen(filename, "w");
    assert( fp != NULL );

    printf("Write array into %s.\n", filename);

    int *h_Array = (int*) malloc( n * sizeof(int) );
    cudaErr = cudaMemcpy(h_Array, d_Array, n*sizeof(int), cudaMemcpyDeviceToHost);
	assert( cudaErr == cudaSuccess );	

    for(int i = 0; i < n; i++)
        fprintf(fp, "%d\n", h_Array[i]);

    fclose(fp);
    free(h_Array);
}

void printDeviceArray(cmpx *d_Array, int n, const char *filename)
{
	cudaError_t cudaErr;
    FILE *fp = fopen(filename, "w");
    assert( fp != NULL );

    printf("Write array into %s.\n", filename);

    int *h_Array = (int*) malloc( n * sizeof(int) );
    cudaErr = cudaMemcpy(h_Array, d_Array, n*sizeof(int), cudaMemcpyDeviceToHost);
	assert( cudaErr == cudaSuccess );	

    for(int i = 0; i < n; i++)
        fprintf(fp, "%d\n", h_Array[i]);

    fclose(fp);
    free(h_Array);
}