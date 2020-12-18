#include "FAME_Internal_Common.h"


void inv3(realCPU* A, realCPU* result)
{
    realCPU det = A[0] * (A[4] * A[8] - A[5] * A[7]) - A[3] * (A[1] * A[8] - A[7] * A[2]) + A[6] * (A[1] * A[5] - A[4] * A[2]);
    realCPU invdet = 1 / det;

    result[0] =  (A[4] * A[8] - A[5] * A[7]) * invdet;
    result[1] = -(A[1] * A[8] - A[2] * A[7]) * invdet;
    result[2] =  (A[1] * A[5] - A[2] * A[4]) * invdet;
    result[3] = -(A[3] * A[8] - A[6] * A[5]) * invdet;
    result[4] =  (A[0] * A[8] - A[6] * A[2]) * invdet;
    result[5] = -(A[0] * A[5] - A[2] * A[3]) * invdet;
    result[6] =  (A[3] * A[7] - A[6] * A[4]) * invdet;
    result[7] = -(A[0] * A[7] - A[1] * A[6]) * invdet;
    result[8] =  (A[0] * A[4] - A[1] * A[3]) * invdet;
}

void inv3_Trans(realCPU* A, realCPU* result)
{
    realCPU det = A[0] * (A[4] * A[8] - A[5] * A[7]) - A[3] * (A[1] * A[8] - A[7] * A[2]) + A[6] * (A[1] * A[5] - A[4] * A[2]);
    realCPU invdet = 1 / det;

    result[0] =  (A[4] * A[8] - A[5] * A[7]) * invdet;
    result[1] = -(A[3] * A[8] - A[6] * A[5]) * invdet;
    result[2] =  (A[3] * A[7] - A[6] * A[4]) * invdet;
    result[3] = -(A[1] * A[8] - A[2] * A[7]) * invdet;
    result[4] =  (A[0] * A[8] - A[6] * A[2]) * invdet;
    result[5] = -(A[0] * A[7] - A[1] * A[6]) * invdet;
    result[6] =  (A[1] * A[5] - A[2] * A[4]) * invdet;
    result[7] = -(A[0] * A[5] - A[2] * A[3]) * invdet;
    result[8] =  (A[0] * A[4] - A[1] * A[3]) * invdet;

}