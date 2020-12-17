#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
typedef double _Complex cmpx;
// 2020-02-19

// Compute the Kronecker Product of two vectors.
void kron_vec(double* vec_C, double* vec_A, int cola, double* vec_B, int colb)
{
    int count = 0;
    for (int i = 0; i < cola; i++ ) {
        for (int j = 0; j < colb; j++ ) {
            vec_C[count] = vec_A[i] * vec_B[j];
            count++;
        }
    }
}

void kron_vec(int* vec_C, int* vec_A, int cola, int* vec_B, int colb)
{
    int count = 0;
    for (int i = 0; i < cola; i++ ) {
        for (int j = 0; j < colb; j++ ) {
            vec_C[count] = vec_A[i] * vec_B[j];
            count++;
        }
    }
}

void kron_vec(cmpx* vec_C, int* vec_A, int cola, cmpx* vec_B, int colb)
{
    int count = 0;
    for (int i = 0; i < cola; i++ ) {
        for (int j = 0; j < colb; j++ ) {
            vec_C[count] = vec_A[i] * vec_B[j];
            count++;
        }
    }
}

void kron_vec(cmpx* vec_C, cmpx* vec_A, int cola, int* vec_B, int colb)
{
    int count = 0;
    for (int i = 0; i < cola; i++ ) {
        for (int j = 0; j < colb; j++ ) {
            vec_C[count] = vec_A[i] * vec_B[j];
            count++;
        }
    }
}

void kron_vec(cmpx* vec_C, cmpx* vec_A, int cola, cmpx* vec_B, int colb)
{
    int count = 0;
    for (int i = 0; i < cola; i++ ) {
        for (int j = 0; j < colb; j++ ) {
            vec_C[count] = vec_A[i] * vec_B[j];
            count++;
        }
    }
}

void kron_vec(cmpx* vec_C, cmpx* vec_A, int cola, double* vec_B, int colb)
{
    int count = 0;
    for (int i = 0; i < cola; i++ ) {
        for (int j = 0; j < colb; j++ ) {
            vec_C[count] = vec_A[i] * vec_B[j];
            count++;
        }
    }
}

void kron_vec(cmpx* vec_C, double* vec_A, int cola, cmpx* vec_B, int colb)
{
    int count = 0;
    for (int i = 0; i < cola; i++ ) {
        for (int j = 0; j < colb; j++ ) {
            vec_C[count] = vec_A[i] * vec_B[j];
            count++;
        }
    }
}

void kron_vec(double* vec_C, double alpha, int* vec_A, int cola, double beta, int* vec_B, int colb)
{
    int count = 0;
    double temp = alpha * beta;
    for (int i = 0; i < cola; i++) {
        for (int j = 0; j < colb; j++) {
            vec_C[count] = temp * ((double)vec_A[i] * (double)vec_B[j]);
            count++;
        }
    }
}