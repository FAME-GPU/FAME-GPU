#include <string>
using namespace std;
#include "FAME_Internal_Common.h"
void mtx_prod(realCPU* ans, realCPU* M1, realCPU* M2, int m, int n, int p);
void mtx_prod(cmpxCPU* ans, cmpxCPU* M1, realCPU* M2, int m, int n, int p);
void mtx_prod(cmpxCPU* ans, realCPU* M1, cmpxCPU* M2, int m, int n, int p);
void mtx_prod(cmpxCPU* ans, cmpxCPU* M1, cmpxCPU* M2, int m, int n, int p);
void mtx_prod(cmpxCPU* ans, int* M1_row, int* M1_col, cmpxCPU* M1_val, cmpxCPU* vec2, int nnz, int m);
void mtx_prod(cmpxCPU* ans, int* M1_row, int* M1_col, cmpxCPU* M1_val, cmpxCPU* vec2, int nnz, int m, std::string flag_CompType);
void mtx_prod(cmpxCPU* ans, int* M1_row, int* M1_col, cmpxCPU* M1_val, cmpxCPU* vec2, int nnz, int m, string flag_CompType1, string flag_CompType2);