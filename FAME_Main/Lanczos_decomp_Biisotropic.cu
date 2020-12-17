#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include <complex.h>
#include <vector>

#include "FAME_Matrix_Vector_Production_Biisotropic_Ar.cuh"
#include "FAME_Matrix_Vector_Production_Biisotropic_Posdef.cuh"
#include "CG_Biiso.cuh"
#include "printDeviceArray.cuh"



int Lanczos_decomp_Biisotropic(
        CULIB_HANDLES cuHandles, 
        FFT_BUFFER    fft_buffer,
        cuDoubleComplex* U, 
        MTX_B mtx_B,
        int Nx,
        int Ny,
        int Nz,
        int Nd,         
        LS  ls,
        double* Lambda_q_sqrt,
        cuDoubleComplex* Pi_Qr,
        cuDoubleComplex* Pi_Pr,
        cuDoubleComplex* Pi_Qrs,
        cuDoubleComplex* Pi_Prs,
        cuDoubleComplex* D_k,
        cuDoubleComplex* D_ks,
        cuDoubleComplex* D_kx,
        cuDoubleComplex* D_ky,
        cuDoubleComplex* D_kz,
        double *T0,
        double *T1, 
        string flag_CompType,
        int loop_start,
        int loop_end,
        PROFILE* Profile)
{
    int max_iter = ls.maxit;
    double tol = ls.tol;
    int i, j;
    int size = 4 * Nd;
    size_t memsize = size * sizeof(cuDoubleComplex);
    double cublas_scale;
    cuDoubleComplex cublas_zcale, alpha_tmp, beta_tmp, Loss;
    cuDoubleComplex *w, *P, *r;
    checkCudaErrors(cudaMalloc((void**)&w, memsize));
    checkCudaErrors(cudaMalloc((void**)&P, memsize*(loop_end+1)));
    checkCudaErrors(cudaMalloc((void**)&r, memsize));
    
    cublasHandle_t cublas_handle = cuHandles.cublas_handle;
    cublasStatus_t cublasErr;

    int iter;   

    for(int ii=0; ii< loop_start; ii++)
    {
        if(flag_CompType == "Simple")
        {
            FAME_Matrix_Vector_Production_Biisotropic_Ar(cuHandles, fft_buffer,U+ii*size, mtx_B, 
            Nx, Ny, Nz, Nd, Pi_Qr, Pi_Pr,Pi_Qrs, Pi_Prs,D_k, D_ks, P+ii*size, Profile);
        }
        else if(flag_CompType == "General")
        {
            FAME_Matrix_Vector_Production_Biisotropic_Ar(cuHandles, fft_buffer, U+ii*size, mtx_B,
            Nx, Ny, Nz, Nd, Pi_Qr, Pi_Pr,Pi_Qrs, Pi_Prs,D_kx, D_ky, D_kz, P+ii*size, Profile);
        }
    }

    if(flag_CompType == "Simple")
    {
        FAME_Matrix_Vector_Production_Biisotropic_Ar(cuHandles, fft_buffer, U+loop_start*size, mtx_B,
             Nx, Ny, Nz, Nd, Pi_Qr, Pi_Pr,Pi_Qrs, Pi_Prs,D_k, D_ks,  r, Profile);
    }
    else if(flag_CompType == "General")
    {
        FAME_Matrix_Vector_Production_Biisotropic_Ar(cuHandles, fft_buffer, U+loop_start*size, mtx_B,
             Nx, Ny, Nz, Nd, Pi_Qr, Pi_Pr,Pi_Qrs, Pi_Prs,D_kx, D_ky, D_kz, r, Profile);
    }
    cublasErr = cublasZdscal_v2( cublas_handle, size, &T1[loop_start], r, 1 );
    assert( cublasErr == CUBLAS_STATUS_SUCCESS );


    for(j=loop_start; j<loop_end; j++)
    {
        //cout<<"j = "<< j <<"================================="<<endl;        

        FAME_Matrix_Vector_Production_Biisotropic_Posdef(cuHandles, U+j*size, mtx_B, Nx, Ny, Nz, Nd, Lambda_q_sqrt,
                    Pi_Qr, Pi_Pr,Pi_Qrs, Pi_Prs,D_k, D_ks, w);
 
        cublas_zcale = make_cuDoubleComplex(- T1[j], 0.0);  
        
        cublasErr = cublasZaxpy_v2(cublas_handle, size, &cublas_zcale, P+(j-1)*size, 1, w, 1);
        assert( cublasErr == CUBLAS_STATUS_SUCCESS );
 
        cublasErr = cublasZdotc_v2(cublas_handle, size, U+j*size, 1, w, 1, &alpha_tmp);
        assert( cublasErr == CUBLAS_STATUS_SUCCESS );

        cublas_scale = 1.0 / T1[j];
        checkCudaErrors(cudaMemcpy(P+j*size, r, memsize, cudaMemcpyDeviceToDevice));
        cublasErr = cublasZdscal_v2(cublas_handle, size, &(cublas_scale), P+j*size, 1);
        assert( cublasErr == CUBLAS_STATUS_SUCCESS );

        cublas_zcale = make_cuDoubleComplex(-alpha_tmp.x, 0.0);
        cublasErr = cublasZaxpy_v2(cublas_handle, size, &cublas_zcale, P+j*size, 1, w, 1);
        assert( cublasErr == CUBLAS_STATUS_SUCCESS );

        checkCudaErrors(cudaMemcpy(r, w, memsize, cudaMemcpyDeviceToDevice));

        /* Full Reorthogonalization */

        for (i = 0; i < j; i++)
        {
            cublasErr = cublasZdotc_v2(cublas_handle, size, U+i*size, 1, r, 1, &Loss );
            assert( cublasErr == CUBLAS_STATUS_SUCCESS );            

            cublas_zcale = make_cuDoubleComplex( -Loss.x, -Loss.y);
            cublasErr = cublasZaxpy_v2(cublas_handle, size, &cublas_zcale, P+i*size, 1, r, 1);
            assert( cublasErr == CUBLAS_STATUS_SUCCESS );
            
            if(i == j)
            {
                T1[j-1] = T1[j-1] + Loss.x;
            }
        }
        T0[j] = alpha_tmp.x + Loss.x;

        struct timespec start, end;
        clock_gettime (CLOCK_REALTIME, &start);

        if(flag_CompType == "Simple")
        {
            iter = CG_Biiso(cuHandles, fft_buffer, mtx_B,tol,max_iter, r, Nx, Ny, Nz, Nd,  D_k, D_ks, Pi_Qr,Pi_Pr, Pi_Qrs,Pi_Prs,  U+(j+1)*size, Profile);

        }
        else if(flag_CompType == "General")
        {
            iter = CG_Biiso(cuHandles, fft_buffer, mtx_B,tol,max_iter, r, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz , Pi_Qr,Pi_Pr, Pi_Qrs,Pi_Prs,  U+(j+1)*size, Profile);
        }
        clock_gettime (CLOCK_REALTIME, &end);
        Profile->ls_iter[Profile->idx] += iter;
        Profile->ls_time[Profile->idx] += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / BILLION;


        cublasErr = cublasZdotc_v2(cublas_handle, size, U+(j+1)*size, 1, r, 1, &beta_tmp);
        assert( cublasErr == CUBLAS_STATUS_SUCCESS );
        T1[j+1] = sqrt(beta_tmp.x);
 
  
        cublas_scale = 1.0 / T1[j+1];
        cublasErr = cublasZdscal_v2( cublas_handle, size, &(cublas_scale), U+(j+1)*size, 1 );
        assert( cublasErr == CUBLAS_STATUS_SUCCESS );
        

    }
    cudaFree(w);cudaFree(P);cudaFree(r);
    return 0;

}

