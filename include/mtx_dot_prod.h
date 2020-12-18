#include "FAME_Internal_Common.h"
void mtx_dot_prod(cmpxCPU* mtx_A, cmpxCPU* mtx_B, int n, int m);
void mtx_dot_prod(realCPU* mtx_A, realCPU* mtx_B, int n, int m);
void mtx_dot_prod(realCPU* mtx_A, cmpxCPU* mtx_B, int n, int m);
void mtx_dot_prod(cmpxCPU* ans, realCPU* mtx_A, cmpxCPU* mtx_B, int n, int m);
void mtx_dot_prod(realCPU* mtx_A, cmpxCPU* mtx_B,cmpxCPU* result, int n, int m);
void mtx_dot_prod(cmpxCPU* mtx_A, cmpxCPU* mtx_B, cmpxCPU* result, int n, int m);