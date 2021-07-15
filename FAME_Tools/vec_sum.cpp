#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "FAME_Internal_Common.h"
// 2020-02-19

int vec_sum(int* vec_x, int len)
{
    int ans = 0;
    for(int i = 0; i < len; i++)
        ans += vec_x[i];
    return ans;
}