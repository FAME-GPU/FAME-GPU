#include "FAME_Internal_Common.h"
void kron_vec(realCPU* vec_C, int* vec_A, int cola, realCPU* vec_B, int colb);
void kron_vec(realCPU* vec_C, realCPU* vec_A, int cola, int* vec_B, int colb);
void kron_vec(realCPU* vec_C, realCPU* vec_A,   int cola, realCPU* vec_B, int colb);
void kron_vec(   int* vec_C,    int* vec_A,   int cola,    int* vec_B, int colb);
void kron_vec(  cmpxCPU* vec_C,    int* vec_A,   int cola,   cmpxCPU* vec_B, int colb);
void kron_vec(  cmpxCPU* vec_C,   cmpxCPU* vec_A,   int cola,    int* vec_B, int colb);
void kron_vec(  cmpxCPU* vec_C,   cmpxCPU* vec_A,   int cola,   cmpxCPU* vec_B, int colb);
void kron_vec(  cmpxCPU* vec_C,   cmpxCPU* vec_A,   int cola, realCPU* vec_B, int colb);
void kron_vec(  cmpxCPU* vec_C, realCPU* vec_A,   int cola,   cmpxCPU* vec_B, int colb);

void kron_vec(realCPU* vec_C, realCPU alpha, int* vec_A, int cola, realCPU beta, int* vec_B, int colb);