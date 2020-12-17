#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include "FAME_Matrix_Vector_Production_Isotropic_invAr.cuh"

static __global__ void Lanczos_init_kernel(cuDoubleComplex *U, const int Asize);

int Lanczos_decomp_gpu( 
    cuDoubleComplex* U,
    double*          T0,
    double*          T1,
    bool             isInit,
    CULIB_HANDLES    cuHandles,
    FFT_BUFFER       fft_buffer,
    LAMBDAS_CUDA     Lambdas_cuda,
    MTX_B            mtx_B,
    LS               ls,
    int Nx, int Ny, int Nz, int Nd, int Nwant, int Nstep,
    string flag_CompType, PROFILE* Profile)
{
    int i, j, loopStart;
    double cublas_scale;
    cuDoubleComplex cublas_zcale, alpha_tmp, Loss;
    int Asize = 2 * Nd;
    
    switch(isInit)
    {
        case 0:
        {
            /* The initial vector */
            dim3 DimBlock(BLOCK_SIZE, 1, 1);
            dim3 DimGrid((Asize - 1) / BLOCK_SIZE + 1, 1, 1);

            Lanczos_init_kernel<<<DimGrid, DimBlock>>>(U, Asize);

            FAME_Matrix_Vector_Production_Isotropic_invAr(U+Asize, U, cuHandles, fft_buffer, Lambdas_cuda, mtx_B,
                                                                     ls, Nx, Ny, Nz, Nd, flag_CompType, Profile);

            cublasZdotc_v2(cuHandles.cublas_handle, Asize, U+Asize, 1, U, 1, &alpha_tmp);

            T0[0] = alpha_tmp.x;

            cublas_zcale = make_cuDoubleComplex(-T0[0], 0.0);
            cublasZaxpy_v2(cuHandles.cublas_handle, Asize, &cublas_zcale, U, 1, U+Asize, 1);

            cublasDznrm2_v2(cuHandles.cublas_handle, Asize, U+Asize, 1, &T1[0]);

            cublas_scale = 1.0 / T1[0];
            cublasZdscal_v2(cuHandles.cublas_handle, Asize, &cublas_scale, U+Asize, 1);

            loopStart = 1;
            break;
        }
        case 1:
        {
            loopStart = Nwant;
            break;
        }
    }

    for(j = loopStart; j < Nstep; j++)
    {
        FAME_Matrix_Vector_Production_Isotropic_invAr(U+Asize*(j+1), U+Asize*j, cuHandles, fft_buffer, Lambdas_cuda, mtx_B,
                                                                               ls, Nx, Ny, Nz, Nd, flag_CompType, Profile);

        cublasZdotc_v2(cuHandles.cublas_handle, Asize, U+Asize*(j+1), 1, U+Asize*j, 1, &alpha_tmp);

        T0[j] = alpha_tmp.x;

        cublas_zcale = make_cuDoubleComplex(-alpha_tmp.x, 0.0);
        cublasZaxpy_v2(cuHandles.cublas_handle, Asize, &cublas_zcale, U+Asize*j, 1, U+Asize*(j+1), 1);

        cublas_zcale = make_cuDoubleComplex(-T1[j - 1], 0.0);
        cublasZaxpy_v2(cuHandles.cublas_handle, Asize, &cublas_zcale, U+Asize*(j-1), 1, U+Asize*(j+1), 1);

        cublasDznrm2_v2(cuHandles.cublas_handle, Asize, U+Asize*(j+1), 1, &T1[j]);
        
        cublas_scale = 1.0 / T1[j];
        cublasZdscal_v2(cuHandles.cublas_handle, Asize, &cublas_scale, U+Asize*(j+1), 1);

        /* Full Reorthogonalization */
        for( i = 0; i <= j; i++)
        {
            cublasZdotc_v2(cuHandles.cublas_handle, Asize, U+Asize*i, 1, U+Asize*(j+1), 1, &Loss);
            
            cublas_zcale = make_cuDoubleComplex(-Loss.x, -Loss.y);
            cublasZaxpy_v2(cuHandles.cublas_handle, Asize, &cublas_zcale, U+Asize*i, 1, U+Asize*(j+1), 1);
        }
    }

    return 0;
}

static __global__ void Lanczos_init_kernel(cuDoubleComplex *U, const int Asize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // e_n
    
    if(idx < Asize - 1)
    {
        U[idx].x = 0.0;
        U[idx].y = 0.0;
    }

    else if(idx == Asize - 1)
    {
        U[idx].x = 1.0;
        U[idx].y = 0.0;
    }
    
/*
    // e_1
    if(idx == 0)
    {
        U[idx].x = 1.0;
        U[idx].y = 0.0;
    }
    else if(idx < Asize)
    {
        U[idx].x = 0.0;
        U[idx].y = 0.0;
    }
    */
}