typedef double _Complex cmpx;
void vec_plus(double* vec_sum, double alpha, double* vec1, double beta, double* vec2, int dim);
void vec_plus(double* vec_sum, double alpha, double* vec1, double beta, int* vec2, int dim);
void vec_plus(double* vec_sum, double alpha, int* vec1, double beta, int* vec2, int dim);
void vec_plus(cmpx* vec_sum, double alpha, cmpx* vec1, double beta, cmpx* vec2, int dim);
void vec_plus(cmpx* vec_sum, cmpx alpha, cmpx* vec1, cmpx beta, cmpx* vec2, int dim);