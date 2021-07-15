#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "FAME_Internal_Common.h"


// Compute the Kronecker Product of two vectors.

void kron_vec(realCPU* vec_C, int* vec_A, int cola, realCPU* vec_B, int colb)
{
    int count = 0;
    for (int i = 0; i < cola; i++ ) {
        for (int j = 0; j < colb; j++ ) {
            vec_C[count] = vec_A[i] * vec_B[j];
            count++;
        }
    }
}

void kron_vec(realCPU* vec_C, realCPU* vec_A, int cola, int* vec_B, int colb)
{
    int count = 0;
    for (int i = 0; i < cola; i++ ) {
        for (int j = 0; j < colb; j++ ) {
            vec_C[count] = vec_A[i] * vec_B[j];
            count++;
        }
    }
}

void kron_vec(realCPU* vec_C, realCPU* vec_A, int cola, realCPU* vec_B, int colb)
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

void kron_vec(cmpxCPU* vec_C, int* vec_A, int cola, cmpxCPU* vec_B, int colb)
{
    int count = 0;
    for (int i = 0; i < cola; i++ ) {
        for (int j = 0; j < colb; j++ ) {
            vec_C[count] = vec_A[i] * vec_B[j];
            count++;
        }
    }
}

void kron_vec(cmpxCPU* vec_C, cmpxCPU* vec_A, int cola, int* vec_B, int colb)
{
    int count = 0;
    for (int i = 0; i < cola; i++ ) {
        for (int j = 0; j < colb; j++ ) {
            vec_C[count] = vec_A[i] * vec_B[j];
            count++;
        }
    }
}

void kron_vec(cmpxCPU* vec_C, cmpxCPU* vec_A, int cola, cmpxCPU* vec_B, int colb)
{
    int count = 0;
    for (int i = 0; i < cola; i++ ) {
        for (int j = 0; j < colb; j++ ) {
            vec_C[count] = vec_A[i] * vec_B[j];
            count++;
        }
    }
}

void kron_vec(cmpxCPU* vec_C, cmpxCPU* vec_A, int cola, realCPU* vec_B, int colb)
{
    int count = 0;
    for (int i = 0; i < cola; i++ ) {
        for (int j = 0; j < colb; j++ ) {
            vec_C[count] = vec_A[i] * vec_B[j];
            count++;
        }
    }
}

void kron_vec(cmpxCPU* vec_C, realCPU* vec_A, int cola, cmpxCPU* vec_B, int colb)
{
    int count = 0;
    for (int i = 0; i < cola; i++ ) {
        for (int j = 0; j < colb; j++ ) {
            vec_C[count] = vec_A[i] * vec_B[j];
            count++;
        }
    }
}

void kron_vec(realCPU* vec_C, realCPU alpha, int* vec_A, int cola, realCPU beta, int* vec_B, int colb)
{
    int count = 0;
    realCPU temp = alpha * beta;
    for (int i = 0; i < cola; i++) {
        for (int j = 0; j < colb; j++) {
            vec_C[count] = temp * ((realCPU)vec_A[i] * (realCPU)vec_B[j]);
            count++;
        }
    }
}