#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include "FAME_Matrix_Vector_Production_Anisotropic_invAr.cuh"

static __global__ void Lanczos_init_kernel(cmpxGPU *U, const int Asize, realCPU tmp);

int Lanczos_decomp_Anisotropic( 
    cmpxGPU* U,
    realGPU*          T0,
    realGPU*          T1,
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
    realGPU cublas_scale;
    cmpxGPU cublas_zcale, alpha_tmp, Loss;
    int Asize = 8 * Nd;
    realGPU tmp = 1.0 / sqrt(Asize);
    
    switch(isInit)
    {
        case 0:
        {
            /* The initial vector */
            dim3 DimBlock(BLOCK_SIZE, 1, 1);
            dim3 DimGrid((Asize - 1) / BLOCK_SIZE + 1, 1, 1);

            Lanczos_init_kernel<<<DimGrid, DimBlock>>>(U, Asize, tmp);
            
            FAME_Matrix_Vector_Production_Anisotropic_invAr(U+Asize, U, cuHandles, fft_buffer, Lambdas_cuda, mtx_B,
                                                                     ls, Nx, Ny, Nz, Nd, flag_CompType, Profile);

            FAME_cublas_dot(cuHandles.cublas_handle, Asize, U+Asize, 1, U, 1, &alpha_tmp);

            T0[0] = alpha_tmp.x;

            cublas_zcale = make_cucmpx(-T0[0], 0.0);
            FAME_cublas_axpy(cuHandles.cublas_handle, Asize, &cublas_zcale, U, 1, U+Asize, 1);

            FAME_cublas_nrm2(cuHandles.cublas_handle, Asize, U+Asize, 1, &T1[0]);

            cublas_scale = 1.0 / T1[0];
            FAME_cublas_dscal(cuHandles.cublas_handle, Asize, &cublas_scale, U+Asize, 1);

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
        FAME_Matrix_Vector_Production_Anisotropic_invAr(U+Asize*(j+1), U+Asize*j, cuHandles, fft_buffer, Lambdas_cuda, mtx_B,
                                                                               ls, Nx, Ny, Nz, Nd, flag_CompType, Profile);

        FAME_cublas_dot(cuHandles.cublas_handle, Asize, U+Asize*(j+1), 1, U+Asize*j, 1, &alpha_tmp);

        T0[j] = alpha_tmp.x;

        cublas_zcale =  make_cucmpx(-alpha_tmp.x, 0.0);
        FAME_cublas_axpy(cuHandles.cublas_handle, Asize, &cublas_zcale, U+Asize*j, 1, U+Asize*(j+1), 1);

        cublas_zcale =  make_cucmpx(-T1[j - 1], 0.0);
        FAME_cublas_axpy(cuHandles.cublas_handle, Asize, &cublas_zcale, U+Asize*(j-1), 1, U+Asize*(j+1), 1);

        FAME_cublas_nrm2(cuHandles.cublas_handle, Asize, U+Asize*(j+1), 1, &T1[j]);
        
        cublas_scale = 1.0 / T1[j];
        FAME_cublas_dscal(cuHandles.cublas_handle, Asize, &cublas_scale, U+Asize*(j+1), 1);

        /* Full Reorthogonalization */
        for( i = 0; i <= j; i++)
        {
            FAME_cublas_dot(cuHandles.cublas_handle, Asize, U+Asize*i, 1, U+Asize*(j+1), 1, &Loss);
            
            cublas_zcale =  make_cucmpx(-Loss.x, -Loss.y);
            FAME_cublas_axpy(cuHandles.cublas_handle, Asize, &cublas_zcale, U+Asize*i, 1, U+Asize*(j+1), 1);
        }
    }

    return 0;
}

static __global__ void Lanczos_init_kernel(cmpxGPU *U, const int Asize, realCPU tmp)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < Asize)
    {
        U[idx].x = tmp;
        U[idx].y = 0.0;
    }

/*
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
    */
    
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