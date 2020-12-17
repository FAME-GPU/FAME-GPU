#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
typedef double _Complex cmpx;
// 2020-02-19

double vec_inner_prod(double* vec1, double* vec2, int len)
{
	double sum = 0;
	for( int i = 0; i < len; i++ )
		sum += vec1[i]*vec2[i];
	return sum;
}

cmpx vec_inner_prod(cmpx* vec_1, cmpx* vec_2, int len)
{
    cmpx ans = 0.0 + 0.0i;
    for(int i = 0; i < len; i++)
        ans += conj(vec_1[i]) * vec_2[i];
    return ans;
}