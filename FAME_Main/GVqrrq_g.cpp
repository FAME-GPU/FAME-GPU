#include "FAME_Internal_Common.h"
#include <iostream>
using namespace std;

void given_rotation(realCPU* c, realCPU* s, realCPU* r, realCPU a, realCPU b);

void GVqrrq_g(realCPU* T0, realCPU* T1, realCPU* T2, realCPU* T3, realCPU* c, realCPU* s, realCPU shift, int n)
{
    int i;
    realCPU a, b, r, R[4];
    
    memcpy(T3, T1, (n-1) * sizeof(realCPU));

    // shift
    for(i = 0; i < n; i++)
        T0[i] -= shift;

    // Q' * T
    for(i = 0; i < n - 1; i++)
    {
        a     = T0[i]; 
        b     = T1[i];

        given_rotation(&c[i], &s[i], &r, a, b);

        T0[i] = r;
        T1[i] = 0.0;
    
        a     = T3[i];
        b     = T0[i + 1];

        T3[i]      = c[i] * a - s[i] * b;
        T0[i + 1]  = s[i] * a + c[i] * b;

        if(i < n - 2)
        {
            T2[i]   = -s[i] * T3[i+1];
            T3[i+1] =  c[i] * T3[i+1];
        }
    }

    for(i = 0; i < n - 1; i++)
    {
        R[0]    = T0[i];
        R[1]    = T3[i];
        R[2]    = T1[i];
        R[3]    = T0[i + 1];

        T0[i]     =  c[i] * R[0] - s[i] * R[1];
        //T3[i]     =  s[i] * R[0] + c[i] * R[1];
        T1[i]     =  c[i] * R[2] - s[i] * R[3];
        T0[i + 1] =  s[i] * R[2] + c[i] * R[3];
    }

    for(i = 0; i < n; i++)
        T0[i] += shift;
}


void given_rotation(realCPU* c, realCPU* s, realCPU* r, realCPU a, realCPU b)
{
    realCPU t, u;
    if(fabs(a) > fabs(b))
    {
        t    = b / a;
        u    = fabs(a) / a * sqrt(1 + t * t);
        c[0] = 1.0 / u;
        s[0] = c[0] * t;
        r[0] = a * u;
        s[0] = -s[0];
    }
    else
    {
        t    = a / b;
        u    = fabs(b) / b * sqrt(1 + t * t);
        s[0] = 1.0 / u;
        c[0] = s[0] * t;
        r[0] = b * u;
        s[0] = -s[0];
    }
}