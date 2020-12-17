#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include "FAME_Matrix_Vector_Production_Isotropic_QBQ.cuh"

int CG(
    cuDoubleComplex* vec_y,
    cuDoubleComplex* rhs,
    CULIB_HANDLES    cuHandles,
    FFT_BUFFER       fft_buffer,
    MTX_B            mtx_B,
    cuDoubleComplex* D_k,
    cuDoubleComplex* D_ks,
    cuDoubleComplex* Pi_Qr,
    cuDoubleComplex* Pi_Qrs,
    int Nx, int Ny, int Nz, int Nd,
    int Maxit, double Tol,
    PROFILE* Profile)
{
    
    int dim = 2 * Nd;
    double res, temp, b;

    cuDoubleComplex a, na, dot, r0, r1;
    cuDoubleComplex one; one.x = 1.0, one.y = 0.0;

    cuDoubleComplex* r  = cuHandles.Nd2_temp2;
    cuDoubleComplex* p  = cuHandles.Nd2_temp3;
    cuDoubleComplex* Ap = cuHandles.Nd2_temp4;

    cudaMemset(vec_y, 0, dim * sizeof(cuDoubleComplex));
    // r = rhs - A * x0 = rhs;
    cublasZcopy_v2(cuHandles.cublas_handle, dim, rhs, 1, r, 1);
    // r1 = dot(r, r);
    cublasZdotc_v2(cuHandles.cublas_handle, dim, r, 1, r, 1, &r1);

    int k = 1;
    while (r1.x > Tol * Tol && k <= Maxit)
    {
        if(k > 1)
        {
            // r0 & r1 are real.
            // p = r + b * p;
            b = r1.x / r0.x;
            cublasZdscal_v2(cuHandles.cublas_handle, dim, &b, p, 1);
            cublasZaxpy_v2(cuHandles.cublas_handle, dim, &one, r, 1, p, 1);
        }
        else
        {
            // p = r;
            cublasZcopy_v2(cuHandles.cublas_handle, dim, r, 1, p, 1);
        }

        // Ap = A * p;
 
        FAME_Matrix_Vector_Production_Isotropic_QBQ(Ap, p, cuHandles, fft_buffer, mtx_B,
                                     D_k, D_ks, Pi_Qr, Pi_Qrs, Nx, Ny, Nz, Nd, Profile);
        
        // dot = dot(p, Ap);
        cublasZdotc_v2(cuHandles.cublas_handle, dim, p, 1, Ap, 1, &dot);

        // a = r1 / dot;
        temp = dot.x * dot.x + dot.y * dot.y;
        a.x =  r1.x * dot.x / temp;
        a.y = -r1.x * dot.y / temp;
        
        // x = a * p + x;

        cublasZaxpy_v2(cuHandles.cublas_handle, dim, &a, p, 1, vec_y, 1);


        // na = -a;
        na.x = -a.x;
        na.y = -a.y;
        // r = -a * Ap + r;
        cublasZaxpy_v2(cuHandles.cublas_handle, dim, &na, Ap, 1, r, 1);

        r0.x = r1.x;
        // r1 = dot(r, r);
        cublasZdotc_v2(cuHandles.cublas_handle, dim, r, 1, r, 1, &r1);
        
        k++;
    }

    res = sqrt(r1.x);
    if(k >= Maxit)
        printf("\033[40;31mCG did not converge when iteration numbers reached LS_MAXIT (%3d) with residual %e.\033[0m\n", Maxit, res);

    return k;
}


int CG(
    cuDoubleComplex* vec_y,
    cuDoubleComplex* rhs,
    CULIB_HANDLES    cuHandles,
    FFT_BUFFER       fft_buffer,
    MTX_B            mtx_B,
    cuDoubleComplex* D_kx,
    cuDoubleComplex* D_ky,
    cuDoubleComplex* D_kz,
    cuDoubleComplex* Pi_Qr,
    cuDoubleComplex* Pi_Qrs,
    int Nx, int Ny, int Nz, int Nd,
    int Maxit, double Tol,
    PROFILE* Profile)
{
    
    int dim = 2 * Nd;
    double res, temp, b;

    cuDoubleComplex a, na, dot, r0, r1;
    cuDoubleComplex one; one.x = 1.0, one.y = 0.0;

    cuDoubleComplex* r  = cuHandles.Nd2_temp2;
    cuDoubleComplex* p  = cuHandles.Nd2_temp3;
    cuDoubleComplex* Ap = cuHandles.Nd2_temp4;

    cudaMemset(vec_y, 0, dim * sizeof(cuDoubleComplex));
    // r = rhs - A * x0 = rhs;
    cublasZcopy_v2(cuHandles.cublas_handle, dim, rhs, 1, r, 1);
    // r1 = dot(r, r);
    cublasZdotc_v2(cuHandles.cublas_handle, dim, r, 1, r, 1, &r1);
    clock_t start1,end1;
    start1 = clock();
    int k = 1;
    while (r1.x > Tol * Tol && k <= Maxit)
    {
        if(k > 1)
        {
            // r0 & r1 are real.
            // p = r + b * p;
            b = r1.x / r0.x;
            cublasZdscal_v2(cuHandles.cublas_handle, dim, &b, p, 1);
            cublasZaxpy_v2(cuHandles.cublas_handle, dim, &one, r, 1, p, 1);
        }
        else
        {
            // p = r;
            cublasZcopy_v2(cuHandles.cublas_handle, dim, r, 1, p, 1);
        }

        // Ap = A * p;
        FAME_Matrix_Vector_Production_Isotropic_QBQ(Ap, p, cuHandles, fft_buffer, mtx_B,
                               D_kx, D_ky, D_kz, Pi_Qr, Pi_Qrs, Nx, Ny, Nz, Nd, Profile);
    
        // dot = dot(p, Ap);
        cublasZdotc_v2(cuHandles.cublas_handle, dim, p, 1, Ap, 1, &dot);

        // a = r1 / dot;
        temp = dot.x * dot.x + dot.y * dot.y;
        a.x =  r1.x * dot.x / temp;
        a.y = -r1.x * dot.y / temp;
        
        // x = a * p + x;
        cublasZaxpy_v2(cuHandles.cublas_handle, dim, &a, p, 1, vec_y, 1);

        // na = -a;
        na.x = -a.x;
        na.y = -a.y;
        // r = -a * Ap + r;
        cublasZaxpy_v2(cuHandles.cublas_handle, dim, &na, Ap, 1, r, 1);

        r0.x = r1.x;
        // r1 = dot(r, r);
        cublasZdotc_v2(cuHandles.cublas_handle, dim, r, 1, r, 1, &r1);

        k++;
    }

    res = sqrt(r1.x);
    if(k >= Maxit)
        printf("\033[40;31mCG did not converge when iteration numbers reached LS_MAXIT (%3d) with residual %e.\033[0m\n", Maxit, res);

    return k;
}
