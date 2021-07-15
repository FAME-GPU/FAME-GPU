#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <assert.h>
#include <math.h>
#include <complex.h>
#include "FAME_Matrix_Vector_Production_Biisotropic_Ar.cuh"
#include "printDeviceArray.cuh"

using namespace std;


int CG_Biiso
	(	CULIB_HANDLES cuHandles, 
		FFT_BUFFER 	fft_buffer,
		MTX_B            mtx_B,
		realGPU Tol, 
		int Maxit, 
		cmpxGPU* rhs, 
		int Nx, int Ny, int Nz, int Nd, 
		cmpxGPU* D_k, 
		cmpxGPU* D_ks, 
		cmpxGPU* Pi_Qr,
		cmpxGPU* Pi_Pr,
		cmpxGPU* Pi_Qrs,		
		cmpxGPU* Pi_Prs,
		cmpxGPU* vec_y,
        PROFILE* Profile)
{
	//cout << "===========================In CG========================" << endl;
	int dim = 4 * Nd;
    realGPU res, temp, b;
    cublasStatus_t cublasStatus;

    cmpxGPU a, na, dot, r0, r1;
    cmpxGPU one; one.x = 1.0, one.y = 0.0;

    cmpxGPU* r  = cuHandles.Nd2_temp2;
    cmpxGPU* p  = cuHandles.Nd2_temp3;
    cmpxGPU* Ap = cuHandles.Nd2_temp4;

    checkCudaErrors(cudaMemset(vec_y, 0, dim * sizeof(cmpxGPU)));
    // r = rhs - A * x0 = rhs;
    FAME_cublas_copy(cuHandles.cublas_handle, dim, rhs, 1, r, 1);
    // r1 = dot(r, r);
    cublasStatus = FAME_cublas_dot(cuHandles.cublas_handle, dim, r, 1, r, 1, &r1);
    assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 


    int k = 1;
    while (r1.x > Tol * Tol && k <= Maxit)
    {
        if(k > 1)
        {
            // r0 & r1 are real.
            // p = r + b * p;
            b = r1.x / r0.x;
            cublasStatus = FAME_cublas_dscal(cuHandles.cublas_handle, dim, &b, p, 1);
            assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 
            cublasStatus = FAME_cublas_axpy(cuHandles.cublas_handle, dim, &one, r, 1, p, 1);
            assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 
        }
        else
        {
            // p = r;
            cublasStatus = FAME_cublas_copy(cuHandles.cublas_handle, dim, r, 1, p, 1);
            assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 
        }

		FAME_Matrix_Vector_Production_Biisotropic_Ar(cuHandles, fft_buffer, p, mtx_B, 
			Nx, Ny, Nz, Nd, Pi_Qr, Pi_Pr, Pi_Qrs, Pi_Prs, D_k, D_ks, Ap, Profile);
        

		// dot = dot(p, Ap);
        cublasStatus = FAME_cublas_dot(cuHandles.cublas_handle, dim, p, 1, Ap, 1, &dot);
        assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 

        // a = r1 / dot;
        temp = dot.x * dot.x + dot.y * dot.y;
        a.x =  r1.x * dot.x / temp;
        a.y = -r1.x * dot.y / temp;

        // x = a * p + x;
        cublasStatus = FAME_cublas_axpy(cuHandles.cublas_handle, dim, &a, p, 1, vec_y, 1);
        assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 

        // na = -a;
        na.x = -a.x;
        na.y = -a.y;
        // r = -a * Ap + r;
        cublasStatus = FAME_cublas_axpy(cuHandles.cublas_handle, dim, &na, Ap, 1, r, 1);
        assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 

        r0.x = r1.x;
        // r1 = dot(r, r);
        cublasStatus = FAME_cublas_dot(cuHandles.cublas_handle, dim, r, 1, r, 1, &r1);
        assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 
        k++;
    }

    res = sqrt(r1.x);

    if(k >= Maxit)
        printf("\033[40;31mCG did not converge when iteration numbers reached LS_MAXIT (%3d) with residual %e.\033[0m\n", Maxit, res);

    return k;
}




int CG_Biiso
	(	CULIB_HANDLES cuHandles, 
		FFT_BUFFER fft_buffer,
		MTX_B            mtx_B,
		realGPU Tol, 
		int Maxit, 
		cmpxGPU* rhs, 
		int Nx, 
		int Ny, 
		int Nz, 
		int Nd, 
		cmpxGPU* D_kx, 
		cmpxGPU* D_ky, 
		cmpxGPU* D_kz, 
		cmpxGPU* Pi_Qr,
		cmpxGPU* Pi_Pr,
		cmpxGPU* Pi_Qrs,		
		cmpxGPU* Pi_Prs,
		cmpxGPU* vec_y,
        PROFILE* Profile)
{
    int dim = 4 * Nd;
    realGPU res, temp, b;
    cublasStatus_t cublasStatus;

    cmpxGPU a, na, dot, r0, r1;
    cmpxGPU one; one.x = 1.0, one.y = 0.0;

    cmpxGPU* r  = cuHandles.Nd2_temp2;
    cmpxGPU* p  = cuHandles.Nd2_temp3;
    cmpxGPU* Ap = cuHandles.Nd2_temp4;

    CHECK_CUDA(cudaMemset(vec_y, 0, dim * sizeof(cmpxGPU)));
    // r = rhs - A * x0 = rhs;
    CHECK_CUBLAS(FAME_cublas_copy(cuHandles.cublas_handle, dim, rhs, 1, r, 1));
    // r1 = dot(r, r);
    CHECK_CUBLAS(FAME_cublas_dot(cuHandles.cublas_handle, dim, r, 1, r, 1, &r1));

    int k = 1;
    while (r1.x > Tol * Tol && k <= Maxit)
    {
        if(k > 1)
        {
            // r0 & r1 are real.
            // p = r + b * p;
            b = r1.x / r0.x;
            CHECK_CUBLAS(FAME_cublas_dscal(cuHandles.cublas_handle, dim, &b, p, 1));
            CHECK_CUBLAS(FAME_cublas_axpy(cuHandles.cublas_handle, dim, &one, r, 1, p, 1));
        }
        else
        {
            // p = r;
            CHECK_CUBLAS(FAME_cublas_copy(cuHandles.cublas_handle, dim, r, 1, p, 1));
        }

		FAME_Matrix_Vector_Production_Biisotropic_Ar(cuHandles, fft_buffer, p,  mtx_B, Nx, Ny, Nz, Nd,
                Pi_Qr, Pi_Pr,Pi_Qrs, Pi_Prs, 
                D_kx, D_ky, D_kz, Ap, Profile);	
        // dot = dot(p, Ap);
        
        CHECK_CUBLAS(FAME_cublas_dot(cuHandles.cublas_handle, dim, p, 1, Ap, 1, &dot));

        // a = r1 / dot;
        temp = dot.x * dot.x + dot.y * dot.y;
        a.x =  r1.x * dot.x / temp;
        a.y = -r1.x * dot.y / temp;
        
        // x = a * p + x;
        CHECK_CUBLAS(FAME_cublas_axpy(cuHandles.cublas_handle, dim, &a, p, 1, vec_y, 1));

        // na = -a;
        na.x = -a.x;
        na.y = -a.y;
        // r = -a * Ap + r;
        CHECK_CUBLAS(FAME_cublas_axpy(cuHandles.cublas_handle, dim, &na, Ap, 1, r, 1));

        r0.x = r1.x;
        // r1 = dot(r, r);
        CHECK_CUBLAS(FAME_cublas_dot(cuHandles.cublas_handle, dim, r, 1, r, 1, &r1));

        k++;
    }

    res = sqrt(r1.x);

    if(k >= Maxit)
        printf("\033[40;31mCG did not converge when iteration numbers reached LS_MAXIT (%3d) with residual %e.\033[0m\n", Maxit, res);

    return k;

}


