#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include <complex.h>
#include <vector>
#include "FAME_Matrix_Vector_Production_Bianisotropic_FdualF.cuh"
#include "FAME_Matrix_Vector_Production_Bianisotropic_i_invSigma.cuh"
#include "CG_FdualF.cuh"
#include "printDeviceArray.cuh"



int Lanczos_decomp_Bianisotropic(
        CULIB_HANDLES cuHandles, 
        FFT_BUFFER    fft_buffer,
        cmpxGPU* U, 
        MTX_B mtx_B,
        int Nx,
        int Ny,
        int Nz,
        int Nd,         
        LS  ls,
        realGPU* Lambda_q_sqrt,
        cmpxGPU* Pi_Qr,
        cmpxGPU* Pi_Pr,
        cmpxGPU* Pi_Qrs,
        cmpxGPU* Pi_Prs,
        cmpxGPU* Pi_Qr_110,
        cmpxGPU* Pi_Pr_110,
        cmpxGPU* Pi_Qrs_110,
        cmpxGPU* Pi_Prs_110,
        cmpxGPU* Pi_Qr_101,
        cmpxGPU* Pi_Pr_101,
        cmpxGPU* Pi_Qrs_101,
        cmpxGPU* Pi_Prs_101,
        cmpxGPU* Pi_Qr_011,
        cmpxGPU* Pi_Pr_011,
        cmpxGPU* Pi_Qrs_011,
        cmpxGPU* Pi_Prs_011,
        cmpxGPU* D_k,
        cmpxGPU* D_ks,
        cmpxGPU* D_kx,
        cmpxGPU* D_ky,
        cmpxGPU* D_kz,
        realGPU *T0,
        realGPU *T1, 
        string flag_CompType,
        int loop_start,
        int loop_end,
        PROFILE* Profile)
{
    int max_iter = ls.maxit;
    realGPU tol = ls.tol;
    int i, j;
    int size = 32 * Nd;
    size_t memsize = size * sizeof(cmpxGPU);
    realGPU cublas_scale;
    cmpxGPU cublas_zcale, alpha_tmp, beta_tmp, Loss;
    cmpxGPU *w, *p, *r;
    checkCudaErrors(cudaMalloc((void**)&w, memsize));
    checkCudaErrors(cudaMalloc((void**)&p, memsize));
    checkCudaErrors(cudaMalloc((void**)&r, memsize));
    
    cublasHandle_t cublas_handle = cuHandles.cublas_handle;
    cublasStatus_t cublasErr;

    int iter;   

    if(flag_CompType == "Simple")
    {
        FAME_Matrix_Vector_Production_Bianisotropic_FdualF(cuHandles, fft_buffer, mtx_B, Nx, Ny, Nz, Nd, 
            Pi_Qr, Pi_Pr, Pi_Qrs, Pi_Prs, Pi_Qr_110, Pi_Pr_110, Pi_Qrs_110, Pi_Prs_110,
            Pi_Qr_101, Pi_Pr_101, Pi_Qrs_101, Pi_Prs_101, Pi_Qr_011, Pi_Pr_011, Pi_Qrs_011, Pi_Prs_011,
            D_k, D_ks, U+loop_start*size, r);
    }
    else if(flag_CompType == "General")
    {
        FAME_Matrix_Vector_Production_Bianisotropic_FdualF(cuHandles, fft_buffer, mtx_B, Nx, Ny, Nz, Nd, 
            Pi_Qr, Pi_Pr, Pi_Qrs, Pi_Prs, Pi_Qr_110, Pi_Pr_110, Pi_Qrs_110, Pi_Prs_110,
            Pi_Qr_101, Pi_Pr_101, Pi_Qrs_101, Pi_Prs_101, Pi_Qr_011, Pi_Pr_011, Pi_Qrs_011, Pi_Prs_011,
            D_kx, D_ky, D_kz, U+loop_start*size, r);
    }
    cublasErr = FAME_cublas_dscal( cublas_handle, size, &T1[loop_start], r, 1 );
    
    
    //cout<<  T1[loop_start]<<endl;
    //getchar();

    for(j=loop_start; j<loop_end; j++)
    {
        //cout<<"j = "<< j <<"================================="<<endl;        

        FAME_Matrix_Vector_Production_Bianisotropic_i_invSigma(cuHandles, Nd, Lambda_q_sqrt, U+j*size, w);
 
        cublas_zcale = make_cucmpx(- T1[j], 0.0);  
        
        if(flag_CompType == "Simple")
        {
            FAME_Matrix_Vector_Production_Bianisotropic_FdualF(cuHandles, fft_buffer, mtx_B, Nx, Ny, Nz, Nd, 
                Pi_Qr, Pi_Pr, Pi_Qrs, Pi_Prs, Pi_Qr_110, Pi_Pr_110, Pi_Qrs_110, Pi_Prs_110,
                Pi_Qr_101, Pi_Pr_101, Pi_Qrs_101, Pi_Prs_101, Pi_Qr_011, Pi_Pr_011, Pi_Qrs_011, Pi_Prs_011,
                D_k, D_ks, U+(j-1)*size, p);
        }
        else if(flag_CompType == "General")
        {
            FAME_Matrix_Vector_Production_Bianisotropic_FdualF(cuHandles, fft_buffer, mtx_B, Nx, Ny, Nz, Nd, 
                Pi_Qr, Pi_Pr, Pi_Qrs, Pi_Prs, Pi_Qr_110, Pi_Pr_110, Pi_Qrs_110, Pi_Prs_110,
                Pi_Qr_101, Pi_Pr_101, Pi_Qrs_101, Pi_Prs_101, Pi_Qr_011, Pi_Pr_011, Pi_Qrs_011, Pi_Prs_011,
                D_kx, D_ky, D_kz, U+(j-1)*size, p);
        }

        cublasErr = FAME_cublas_axpy(cublas_handle, size, &cublas_zcale, p, 1, w, 1);
        assert( cublasErr == CUBLAS_STATUS_SUCCESS );
 
        cublasErr = FAME_cublas_dot(cublas_handle, size, U+j*size, 1, w, 1, &alpha_tmp);
        assert( cublasErr == CUBLAS_STATUS_SUCCESS );

        cublas_scale = 1.0 / T1[j];
        checkCudaErrors(cudaMemcpy(p, r, memsize, cudaMemcpyDeviceToDevice));
        cublasErr = FAME_cublas_dscal(cublas_handle, size, &(cublas_scale), p, 1);
        assert( cublasErr == CUBLAS_STATUS_SUCCESS );

        cublas_zcale = make_cucmpx(-alpha_tmp.x, 0.0);
        cublasErr = FAME_cublas_axpy(cublas_handle, size, &cublas_zcale, p, 1, w, 1);
        assert( cublasErr == CUBLAS_STATUS_SUCCESS );

        checkCudaErrors(cudaMemcpy(r, w, memsize, cudaMemcpyDeviceToDevice));
           

        /* Full Reorthogonalization */
        for (i = 0; i < j; i++)
        {
            cublasErr = FAME_cublas_dot(cublas_handle, size, U+i*size, 1, r, 1, &Loss );
            assert( cublasErr == CUBLAS_STATUS_SUCCESS );            

            if(flag_CompType == "Simple")
            {
                FAME_Matrix_Vector_Production_Bianisotropic_FdualF(cuHandles, fft_buffer, mtx_B, Nx, Ny, Nz, Nd, 
                    Pi_Qr, Pi_Pr, Pi_Qrs, Pi_Prs, Pi_Qr_110, Pi_Pr_110, Pi_Qrs_110, Pi_Prs_110,
                    Pi_Qr_101, Pi_Pr_101, Pi_Qrs_101, Pi_Prs_101, Pi_Qr_011, Pi_Pr_011, Pi_Qrs_011, Pi_Prs_011,
                    D_k, D_ks, U+i*size, p);
            }
            else if(flag_CompType == "General")
            {
                FAME_Matrix_Vector_Production_Bianisotropic_FdualF(cuHandles, fft_buffer, mtx_B, Nx, Ny, Nz, Nd, 
                    Pi_Qr, Pi_Pr, Pi_Qrs, Pi_Prs, Pi_Qr_110, Pi_Pr_110, Pi_Qrs_110, Pi_Prs_110,
                    Pi_Qr_101, Pi_Pr_101, Pi_Qrs_101, Pi_Prs_101, Pi_Qr_011, Pi_Pr_011, Pi_Qrs_011, Pi_Prs_011,
                    D_kx, D_ky, D_kz, U+i*size, p);
            }

            cublas_zcale = make_cucmpx( -Loss.x, -Loss.y);
            cublasErr = FAME_cublas_axpy(cublas_handle, size, &cublas_zcale, p, 1, r, 1);
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
            iter = CG_FdualF(r, U+(j+1)*size, cuHandles, fft_buffer, mtx_B, Nx, Ny, Nz, Nd, max_iter, tol, Pi_Qr, Pi_Pr, Pi_Qrs, Pi_Prs, 
                Pi_Qr_110, Pi_Pr_110, Pi_Qrs_110, Pi_Prs_110, Pi_Qr_101, Pi_Pr_101, Pi_Qrs_101, Pi_Prs_101, Pi_Qr_011, Pi_Pr_011, Pi_Qrs_011, Pi_Prs_011,
                D_k, D_ks);

        }
        else if(flag_CompType == "General")
        {
            iter = CG_FdualF(r, U+(j+1)*size, cuHandles, fft_buffer, mtx_B, Nx, Ny, Nz, Nd, max_iter, tol, Pi_Qr, Pi_Pr, Pi_Qrs, Pi_Prs, 
                Pi_Qr_110, Pi_Pr_110, Pi_Qrs_110, Pi_Prs_110, Pi_Qr_101, Pi_Pr_101, Pi_Qrs_101, Pi_Prs_101, Pi_Qr_011, Pi_Pr_011, Pi_Qrs_011, Pi_Prs_011,
                D_kx, D_ky, D_kz);  
        }
        clock_gettime (CLOCK_REALTIME, &end);
        Profile->ls_iter[Profile->idx] += iter;
        Profile->ls_time[Profile->idx] += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / BILLION;


        cublasErr = FAME_cublas_dot(cublas_handle, size, U+(j+1)*size, 1, r, 1, &beta_tmp);
        assert( cublasErr == CUBLAS_STATUS_SUCCESS );
        T1[j+1] = sqrt(beta_tmp.x);
 
  
        cublas_scale = 1.0 / T1[j+1];
        cublasErr = FAME_cublas_dscal( cublas_handle, size, &(cublas_scale), U+(j+1)*size, 1 );
        assert( cublasErr == CUBLAS_STATUS_SUCCESS );
        

    }

    cudaFree(w);cudaFree(p);cudaFree(r);

    return 0;

}

