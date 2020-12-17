#include "FAME_Internal_Common.h"

int FAME_Malloc_mtx_C(MTX_C* mtx_C, int N)
{
	int N2 = N * 2;
	int N12 = N * 12;
	size_t size;

	size = N2 * sizeof(int);
    mtx_C->C1_r = (int*) malloc(size);
    mtx_C->C1_c = (int*) malloc(size);
	mtx_C->C2_r = (int*) malloc(size);
	mtx_C->C2_c = (int*) malloc(size);
	mtx_C->C3_r = (int*) malloc(size);
	mtx_C->C3_c = (int*) malloc(size);

	size = N2 * sizeof(cmpx);
	mtx_C->C1_v = (cmpx*) malloc(size);
	mtx_C->C2_v = (cmpx*) malloc(size);
	mtx_C->C3_v = (cmpx*) malloc(size);

	size = N12 * sizeof(int);
	mtx_C->C_r = (int*)  malloc(size);
	mtx_C->C_c = (int*)  malloc(size);
	mtx_C->C_v = (cmpx*) malloc(N12 * sizeof(cmpx));

	return 0;
}