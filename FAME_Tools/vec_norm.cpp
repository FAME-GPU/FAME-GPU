#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
typedef double _Complex cmpx;
// 2020-02-19

double vec_norm(cmpx* vec_x, int len)
{
    cmpx ans = 0.0 + 0.0i;
    for(int i = 0; i < len; i++)
        ans += vec_x[i] * conj(vec_x[i]);
    return sqrt(creal(ans));
}

double vec_norm(double* vec_x, int len)
{
    double ans = 0.0;
    for(int i = 0; i < len; i++)
        ans += vec_x[i] * vec_x[i];
    return sqrt(ans);
}