#include <string>
using namespace std;
typedef double _Complex cmpx;
void mtx_prod(double* ans, double* M1, double* M2, int m, int n, int p);
void mtx_prod(cmpx* ans, cmpx* M1, double* M2, int m, int n, int p);
void mtx_prod(cmpx* ans, double* M1, cmpx* M2, int m, int n, int p);
void mtx_prod(cmpx* ans, cmpx* M1, cmpx* M2, int m, int n, int p);
void mtx_prod(cmpx* ans, int* M1_row, int* M1_col, cmpx* M1_val, cmpx* vec2, int nnz, int m);
void mtx_prod(cmpx* ans, int* M1_row, int* M1_col, cmpx* M1_val, cmpx* vec2, int nnz, int m, std::string flag_CompType);
