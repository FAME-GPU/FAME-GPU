#include "FAME_Internal_Common.h"
void mtx_trans_conj(cmpxCPU* mtx_A, cmpxCPU* mtx_B, int n, int m);
void mtx_trans_conj(int* mtx_A_r, int* mtx_A_c, cmpxCPU* mtx_A_v, 
	                int* mtx_B_r, int* mtx_B_c, cmpxCPU* mtx_B_v, 
	                int  nnz);