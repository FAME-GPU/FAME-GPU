#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "FAME_Internal_Common.h"
// 2020-02-19

realCPU vec_norm(cmpxCPU* vec_x, int len)
{
    cmpxCPU ans = 0.0 + 0.0j;
    for(int i = 0; i < len; i++)
        ans += vec_x[i] * conj(vec_x[i]);
    return sqrt(creal(ans));
}

realCPU vec_norm(realCPU* vec_x, int len)
{
    realCPU ans = 0.0;
    for(int i = 0; i < len; i++)
        ans += vec_x[i] * vec_x[i];
    return sqrt(ans);
}