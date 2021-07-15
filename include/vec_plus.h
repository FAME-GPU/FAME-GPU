#include "FAME_Internal_Common.h"
void vec_plus(realCPU alpha, realCPU* vec1, realCPU beta, realCPU* vec2, int dim);
void vec_plus(realCPU* vec_sum, realCPU alpha, realCPU* vec1, realCPU beta, realCPU* vec2, int dim);
void vec_plus(realCPU* vec_sum, realCPU alpha, realCPU* vec1, realCPU beta, int* vec2, int dim);
void vec_plus(realCPU* vec_sum, realCPU alpha, int* vec1, realCPU beta, int* vec2, int dim);
void vec_plus(cmpxCPU* vec_sum, realCPU alpha, cmpxCPU* vec1, realCPU beta, cmpxCPU* vec2, int dim);
void vec_plus(cmpxCPU* vec_sum, cmpxCPU alpha, cmpxCPU* vec1, cmpxCPU beta, cmpxCPU* vec2, int dim);