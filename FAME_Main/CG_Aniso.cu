#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include "FAME_Matrix_Vector_Production_Anisotropic_QBQ.cuh"

int CG_Aniso(
    cmpxGPU* vec_y,
    cmpxGPU* rhs,
    CULIB_HANDLES    cuHandles,
    FFT_BUFFER       fft_buffer,
    MTX_B            mtx_B,
    cmpxGPU* D_k,
    cmpxGPU* D_ks,
    cmpxGPU* Pi_Qr,
    cmpxGPU* Pi_Qrs,
    cmpxGPU* Pi_Qr_110,
    cmpxGPU* Pi_Qrs_110,
    cmpxGPU* Pi_Qr_101,
    cmpxGPU* Pi_Qrs_101,
    cmpxGPU* Pi_Qr_011,
    cmpxGPU* Pi_Qrs_011,
    int Nx, int Ny, int Nz, int Nd,
    int Maxit, realGPU Tol,
    PROFILE* Profile)
{
    
    int dim = 8 * Nd;
    realGPU res, temp, b;

    cmpxGPU a, na, dot, r0, r1;
    cmpxGPU one; one.x = 1.0, one.y = 0.0;

    cmpxGPU* r  = cuHandles.Nd2_temp2;
    cmpxGPU* p  = cuHandles.Nd2_temp3;
    cmpxGPU* Ap = cuHandles.Nd2_temp4;

    cudaMemset(vec_y, 0, dim * sizeof(cmpxGPU));
    // r = rhs - A * x0 = rhs;
    FAME_cublas_copy(cuHandles.cublas_handle, dim, rhs, 1, r, 1);
    // r1 = dot(r, r);
    FAME_cublas_dot(cuHandles.cublas_handle, dim, r, 1, r, 1, &r1);

    int k = 1;
    while (r1.x > Tol * Tol && k <= Maxit)
    {
        if(k > 1)
        {
            // r0 & r1 are real.
            // p = r + b * p;
            b = r1.x / r0.x;
            FAME_cublas_dscal(cuHandles.cublas_handle, dim, &b, p, 1);
            FAME_cublas_axpy(cuHandles.cublas_handle, dim, &one, r, 1, p, 1);
        }
        else
        {
            // p = r;
            FAME_cublas_copy(cuHandles.cublas_handle, dim, r, 1, p, 1);
        }

        // Ap = A * p;
                                     
        FAME_Matrix_Vector_Production_Anisotropic_QBQ(Ap, p, cuHandles, fft_buffer, mtx_B,
                D_k, D_ks, Pi_Qr, Pi_Qrs, Pi_Qr_110, Pi_Qrs_110, Pi_Qr_101, Pi_Qrs_101, Pi_Qr_011, Pi_Qrs_011, Nx, Ny, Nz, Nd);
        
        // dot = dot(p, Ap);
        FAME_cublas_dot(cuHandles.cublas_handle, dim, p, 1, Ap, 1, &dot);

        // a = r1 / dot;
        temp = dot.x * dot.x + dot.y * dot.y;
        a.x =  r1.x * dot.x / temp;
        a.y = -r1.x * dot.y / temp;
        
        // x = a * p + x;

        FAME_cublas_axpy(cuHandles.cublas_handle, dim, &a, p, 1, vec_y, 1);


        // na = -a;
        na.x = -a.x;
        na.y = -a.y;
        // r = -a * Ap + r;
        FAME_cublas_axpy(cuHandles.cublas_handle, dim, &na, Ap, 1, r, 1);

        r0.x = r1.x;
        // r1 = dot(r, r);
        FAME_cublas_dot(cuHandles.cublas_handle, dim, r, 1, r, 1, &r1);
        
        k++;
    }

    res = sqrt(r1.x);
    // printf("%d %e.\n", k, res);
    if(k >= Maxit)
        printf("\033[40;31mCG did not converge when iteration numbers reached LS_MAXIT (%3d) with residual %e.\033[0m\n", Maxit, res);

    return k;
}


int CG_Aniso(
    cmpxGPU* vec_y,
    cmpxGPU* rhs,
    CULIB_HANDLES    cuHandles,
    FFT_BUFFER       fft_buffer,
    MTX_B            mtx_B,
    cmpxGPU* D_kx,
    cmpxGPU* D_ky,
    cmpxGPU* D_kz,
    cmpxGPU* Pi_Qr,
    cmpxGPU* Pi_Qrs,
    cmpxGPU* Pi_Qr_110,
    cmpxGPU* Pi_Qrs_110,
    cmpxGPU* Pi_Qr_101,
    cmpxGPU* Pi_Qrs_101,
    cmpxGPU* Pi_Qr_011,
    cmpxGPU* Pi_Qrs_011,
    int Nx, int Ny, int Nz, int Nd,
    int Maxit, realGPU Tol,
    PROFILE* Profile)
{
    
    int dim = 8 * Nd;
    realGPU res, temp, b;

    cmpxGPU a, na, dot, r0, r1;
    cmpxGPU one; one.x = 1.0, one.y = 0.0;

    cmpxGPU* r  = cuHandles.Nd2_temp2;
    cmpxGPU* p  = cuHandles.Nd2_temp3;
    cmpxGPU* Ap = cuHandles.Nd2_temp4;

    cudaMemset(vec_y, 0, dim * sizeof(cmpxGPU));
    // r = rhs - A * x0 = rhs;
    FAME_cublas_copy(cuHandles.cublas_handle, dim, rhs, 1, r, 1);
    // r1 = dot(r, r);
    FAME_cublas_dot(cuHandles.cublas_handle, dim, r, 1, r, 1, &r1);

    int k = 1;
    while (r1.x > Tol * Tol && k <= Maxit)
    {
        if(k > 1)
        {
            // r0 & r1 are real.
            // p = r + b * p;
            b = r1.x / r0.x;
            FAME_cublas_dscal(cuHandles.cublas_handle, dim, &b, p, 1);
            FAME_cublas_axpy(cuHandles.cublas_handle, dim, &one, r, 1, p, 1);
        }
        else
        {
            // p = r;
            FAME_cublas_copy(cuHandles.cublas_handle, dim, r, 1, p, 1);
        }

        // Ap = A * p;       
        FAME_Matrix_Vector_Production_Anisotropic_QBQ(Ap, p, cuHandles, fft_buffer, mtx_B,
                D_kx, D_ky, D_kz, Pi_Qr, Pi_Qrs, Pi_Qr_110, Pi_Qrs_110, Pi_Qr_101, Pi_Qrs_101, Pi_Qr_011, Pi_Qrs_011, Nx, Ny, Nz, Nd);
    
        // dot = dot(p, Ap);
        FAME_cublas_dot(cuHandles.cublas_handle, dim, p, 1, Ap, 1, &dot);

        // a = r1 / dot;
        temp = dot.x * dot.x + dot.y * dot.y;
        a.x =  r1.x * dot.x / temp;
        a.y = -r1.x * dot.y / temp;
        
        // x = a * p + x;
        FAME_cublas_axpy(cuHandles.cublas_handle, dim, &a, p, 1, vec_y, 1);

        // na = -a;
        na.x = -a.x;
        na.y = -a.y;
        // r = -a * Ap + r;
        FAME_cublas_axpy(cuHandles.cublas_handle, dim, &na, Ap, 1, r, 1);

        r0.x = r1.x;
        // r1 = dot(r, r);
        FAME_cublas_dot(cuHandles.cublas_handle, dim, r, 1, r, 1, &r1);

        k++;
    }

    res = sqrt(r1.x);
    if(k >= Maxit)
        printf("\033[40;31mCG did not converge when iteration numbers reached LS_MAXIT (%3d) with residual %e.\033[0m\n", Maxit, res);

    return k;
}
