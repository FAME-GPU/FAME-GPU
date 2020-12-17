#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include <complex.h>
#include <vector>

#include "printDeviceArray.cuh"

#define BLOCK_SIZE 256
void givens(double* c, double* s, double a, double b);

int Lanczos_LockPurge(CULIB_HANDLES cuHandles,  LANCZOS_BUFFER* lBuffer, cuDoubleComplex* ev, 
    int length, int Nstep, int size)
{   
    int     j, i;
    double c  = 0;
    double s  = 0;
    double zj,zjj;
    double temp[4],gamma;
    dim3 DimBlock( BLOCK_SIZE, 1, 1);
    dim3 DimGrid( (size-1)/BLOCK_SIZE +1, 1, 1);

    double* LT0 = lBuffer->LT0;
    double* beta = lBuffer->T2;
    cmpx *z = lBuffer->z;
 
    
    for (i=0;i<length;i++)
        beta[i]=0;
    cublasHandle_t cublas_handle = cuHandles.cublas_handle;
    cublasStatus_t cublasStatus;

    zj=creal(z[Nstep-1]);

    for (j=0; j<length; j++)
    {
        zjj=creal(z[(j+2)*Nstep-1]);
        givens(&c, &s, zjj, zj);
        zj=-s*zj+c*zjj;        

        temp[0]=c*LT0[j]+s*beta[j];
        temp[1]=c*beta[j]+s*LT0[j+1];
        temp[2]=-s*LT0[j]+c*beta[j];
        temp[3]=-s*beta[j]+c*LT0[j+1];


        LT0[j]=c*temp[0]+s*temp[1];
        beta[j]=-s*temp[0]+c*temp[1];
        LT0[j+1]=-s*temp[2]+c*temp[3];


        if (j>0)
        {
            gamma=-s*beta[j-1];          
            beta[j-1]=c*beta[j-1];
        }

        //ev=ev*G';
        cublasStatus=cublasZdrot_v2(cublas_handle, size, ev+j*size, 1, ev+(j+1)*size, 1, &c, &s);
        assert( cublasStatus == CUBLAS_STATUS_SUCCESS );

        if(j>0)
        {
            for (i=j;i>0;i--)
            {
                givens(&c, &s, beta[i], gamma);
                temp[0]=c*LT0[i-1]+s*beta[i-1];
                temp[1]=c*beta[i-1]+s*LT0[i];
                temp[2]=-s*LT0[i-1]+c*beta[i-1];
                temp[3]=-s*beta[i-1]+c*LT0[i];
        
                LT0[i-1]=temp[0]*c+s*temp[1];
                beta[i-1]=-temp[0]*s+c*temp[1];
                LT0[i]=-temp[2]*s+c*temp[3];
                
                beta[i]=-s*gamma+c*beta[i];

                if(i>1)
                {
                    gamma=-s*beta[i-2];
                    beta[i-2]=c*beta[i-2];
                }

                cublasStatus=cublasZdrot_v2(cublas_handle, size, ev+(i-1)*size, 1, ev+i*size, 1, &c, &s);
                assert( cublasStatus == CUBLAS_STATUS_SUCCESS );
            }

        }
    } 
    beta[length] = zj;

    return 0;

}


void givens(double* c, double* s, double a, double b)
{
    double t;
    if (fabs(b) < 1e-15)
    {
        c[0]=1;s[0]=0;
    }
    else if(fabs(a) > fabs(b))
        {
            t    = b / a;
            c[0] = 1.0 / sqrt(1 + t * t);
            s[0] = c[0] * t;
            s[0] = -s[0];
        }
    else
    {
        t    = a / b;
        s[0] = 1.0 /  sqrt(1 + t * t);
        c[0] = s[0] * t;
        s[0] = -s[0];
    }
}