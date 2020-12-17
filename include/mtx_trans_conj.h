typedef double _Complex cmpx;
void mtx_trans_conj(cmpx* mtx_A, cmpx* mtx_B, int n, int m);
void mtx_trans_conj(int* mtx_A_r, int* mtx_A_c, cmpx* mtx_A_v, 
	                int* mtx_B_r, int* mtx_B_c, cmpx* mtx_B_v, 
	                int  nnz);