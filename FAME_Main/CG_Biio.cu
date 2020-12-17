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
		double Tol, 
		int Maxit, 
		cuDoubleComplex* rhs, 
		int Nx, int Ny, int Nz, int Nd, 
		cuDoubleComplex* D_k, 
		cuDoubleComplex* D_ks, 
		cuDoubleComplex* Pi_Qr,
		cuDoubleComplex* Pi_Pr,
		cuDoubleComplex* Pi_Qrs,		
		cuDoubleComplex* Pi_Prs,
		cuDoubleComplex* vec_y,
        PROFILE* Profile)
{
	//cout << "===========================In CG========================" << endl;
	int dim = 4 * Nd;
    double res, temp, b;
    cublasStatus_t cublasStatus;

    cuDoubleComplex a, na, dot, r0, r1;
    cuDoubleComplex one; one.x = 1.0, one.y = 0.0;

    cuDoubleComplex* r  = cuHandles.Nd2_temp2;
    cuDoubleComplex* p  = cuHandles.Nd2_temp3;
    cuDoubleComplex* Ap = cuHandles.Nd2_temp4;

    checkCudaErrors(cudaMemset(vec_y, 0, dim * sizeof(cuDoubleComplex)));
    // r = rhs - A * x0 = rhs;
    cublasZcopy_v2(cuHandles.cublas_handle, dim, rhs, 1, r, 1);
    // r1 = dot(r, r);
    cublasStatus = cublasZdotc_v2(cuHandles.cublas_handle, dim, r, 1, r, 1, &r1);
    assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 


    int k = 1;
    while (r1.x > Tol * Tol && k <= Maxit)
    {
        if(k > 1)
        {
            // r0 & r1 are real.
            // p = r + b * p;
            b = r1.x / r0.x;
            cublasStatus = cublasZdscal_v2(cuHandles.cublas_handle, dim, &b, p, 1);
            assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 
            cublasStatus = cublasZaxpy_v2(cuHandles.cublas_handle, dim, &one, r, 1, p, 1);
            assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 
        }
        else
        {
            // p = r;
            cublasStatus = cublasZcopy_v2(cuHandles.cublas_handle, dim, r, 1, p, 1);
            assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 
        }

		FAME_Matrix_Vector_Production_Biisotropic_Ar(cuHandles, fft_buffer, p, mtx_B, 
			Nx, Ny, Nz, Nd, Pi_Qr, Pi_Pr, Pi_Qrs, Pi_Prs, D_k, D_ks, Ap, Profile);
        

		// dot = dot(p, Ap);
        cublasStatus = cublasZdotc_v2(cuHandles.cublas_handle, dim, p, 1, Ap, 1, &dot);
        assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 

        // a = r1 / dot;
        temp = dot.x * dot.x + dot.y * dot.y;
        a.x =  r1.x * dot.x / temp;
        a.y = -r1.x * dot.y / temp;

        // x = a * p + x;
        cublasStatus = cublasZaxpy_v2(cuHandles.cublas_handle, dim, &a, p, 1, vec_y, 1);
        assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 

        // na = -a;
        na.x = -a.x;
        na.y = -a.y;
        // r = -a * Ap + r;
        cublasStatus = cublasZaxpy_v2(cuHandles.cublas_handle, dim, &na, Ap, 1, r, 1);
        assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 

        r0.x = r1.x;
        // r1 = dot(r, r);
        cublasStatus = cublasZdotc_v2(cuHandles.cublas_handle, dim, r, 1, r, 1, &r1);
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
		double Tol, 
		int Maxit, 
		cuDoubleComplex* rhs, 
		int Nx, 
		int Ny, 
		int Nz, 
		int Nd, 
		cuDoubleComplex* D_kx, 
		cuDoubleComplex* D_ky, 
		cuDoubleComplex* D_kz, 
		cuDoubleComplex* Pi_Qr,
		cuDoubleComplex* Pi_Pr,
		cuDoubleComplex* Pi_Qrs,		
		cuDoubleComplex* Pi_Prs,
		cuDoubleComplex* vec_y,
        PROFILE* Profile)
{
    int dim = 4 * Nd;
    double res, temp, b;
    cublasStatus_t cublasStatus;

    cuDoubleComplex a, na, dot, r0, r1;
    cuDoubleComplex one; one.x = 1.0, one.y = 0.0;

    cuDoubleComplex* r  = cuHandles.Nd2_temp2;
    cuDoubleComplex* p  = cuHandles.Nd2_temp3;
    cuDoubleComplex* Ap = cuHandles.Nd2_temp4;

    checkCudaErrors(cudaMemset(vec_y, 0, dim * sizeof(cuDoubleComplex)));
    // r = rhs - A * x0 = rhs;
    cublasStatus = cublasZcopy_v2(cuHandles.cublas_handle, dim, rhs, 1, r, 1);
    assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 
    // r1 = dot(r, r);
    cublasStatus = cublasZdotc_v2(cuHandles.cublas_handle, dim, r, 1, r, 1, &r1);
    assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 

    int k = 1;
    while (r1.x > Tol * Tol && k <= Maxit)
    {
        if(k > 1)
        {
            // r0 & r1 are real.
            // p = r + b * p;
            b = r1.x / r0.x;
            cublasStatus = cublasZdscal_v2(cuHandles.cublas_handle, dim, &b, p, 1);
            assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 
            cublasStatus = cublasZaxpy_v2(cuHandles.cublas_handle, dim, &one, r, 1, p, 1);
            assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 
        }
        else
        {
            // p = r;
            cublasStatus = cublasZcopy_v2(cuHandles.cublas_handle, dim, r, 1, p, 1);
            assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 
        }

		FAME_Matrix_Vector_Production_Biisotropic_Ar(cuHandles, fft_buffer, p,  mtx_B, Nx, Ny, Nz, Nd,
                Pi_Qr, Pi_Pr,Pi_Qrs, Pi_Prs, 
                D_kx, D_ky, D_kz, Ap, Profile);	
        // dot = dot(p, Ap);
        
        cublasStatus = cublasZdotc_v2(cuHandles.cublas_handle, dim, p, 1, Ap, 1, &dot);
        assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 

        // a = r1 / dot;
        temp = dot.x * dot.x + dot.y * dot.y;
        a.x =  r1.x * dot.x / temp;
        a.y = -r1.x * dot.y / temp;
        
        // x = a * p + x;
        cublasStatus = cublasZaxpy_v2(cuHandles.cublas_handle, dim, &a, p, 1, vec_y, 1);
        assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 

        // na = -a;
        na.x = -a.x;
        na.y = -a.y;
        // r = -a * Ap + r;
        cublasStatus = cublasZaxpy_v2(cuHandles.cublas_handle, dim, &na, Ap, 1, r, 1);
        assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 

        r0.x = r1.x;
        // r1 = dot(r, r);
        cublasStatus = cublasZdotc_v2(cuHandles.cublas_handle, dim, r, 1, r, 1, &r1);
        assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 
        k++;
    }

    res = sqrt(r1.x);

    if(k >= Maxit)
        printf("\033[40;31mCG did not converge when iteration numbers reached LS_MAXIT (%3d) with residual %e.\033[0m\n", Maxit, res);

    return k;

}


