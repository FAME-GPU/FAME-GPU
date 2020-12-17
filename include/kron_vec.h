typedef double _Complex cmpx;

void kron_vec(double* vec_C, double* vec_A,   int cola, double* vec_B, int colb);
void kron_vec(   int* vec_C,    int* vec_A,   int cola,    int* vec_B, int colb);
void kron_vec(  cmpx* vec_C,    int* vec_A,   int cola,   cmpx* vec_B, int colb);
void kron_vec(  cmpx* vec_C,   cmpx* vec_A,   int cola,    int* vec_B, int colb);
void kron_vec(  cmpx* vec_C,   cmpx* vec_A,   int cola,   cmpx* vec_B, int colb);
void kron_vec(  cmpx* vec_C,   cmpx* vec_A,   int cola, double* vec_B, int colb);
void kron_vec(  cmpx* vec_C, double* vec_A,   int cola,   cmpx* vec_B, int colb);

void kron_vec(double* vec_C, double alpha, int* vec_A, int cola, double beta, int* vec_B, int colb);