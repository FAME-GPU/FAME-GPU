typedef double _Complex cmpx;
void mtx_dot_prod(cmpx* mtx_A, cmpx* mtx_B, int n, int m);
void mtx_dot_prod(double* mtx_A, double* mtx_B, int n, int m);
void mtx_dot_prod(double* mtx_A, cmpx* mtx_B, int n, int m);
void mtx_dot_prod(cmpx* ans, double* mtx_A, cmpx* mtx_B, int n, int m);
void mtx_dot_prod(double* mtx_A, cmpx* mtx_B,cmpx* result, int n, int m);
void mtx_dot_prod(cmpx* mtx_A, cmpx* mtx_B, cmpx* result, int n, int m);