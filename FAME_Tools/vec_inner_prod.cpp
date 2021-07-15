#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "FAME_Internal_Common.h"
// 2020-02-19

realCPU vec_inner_prod(realCPU* vec1, realCPU* vec2, int len)
{
	realCPU sum = 0;
	for( int i = 0; i < len; i++ )
		sum += vec1[i]*vec2[i];
	return sum;
}

cmpxCPU vec_inner_prod(cmpxCPU* vec_1, cmpxCPU* vec_2, int len)
{
    cmpxCPU ans = 0.0 + 0.0j;
    for(int i = 0; i < len; i++)
        ans += conj(vec_1[i]) * vec_2[i];
    return ans;
}