#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include "Lanczos_decomp_gpu.cuh"
#include "Lanczos_LockPurge.cuh"
#include "cuda_profiler_api.h"
#include <lapacke.h>

int Lanczos_Isotropic( 
    double*          Freq_array, 
    cuDoubleComplex* ev,
    CULIB_HANDLES    cuHandles,
    LANCZOS_BUFFER   lBuffer,
    FFT_BUFFER       fft_buffer,
    LAMBDAS_CUDA     Lambdas_cuda,
    MTX_B            mtx_B,
    ES               es,
    LS               ls,
    int Nx, int Ny, int Nz, int Nd,
    string flag_CompType, PROFILE* Profile)
{
    cout << "In Lanczos_Isotropic" << endl;
    cublasStatus_t cublasStatus;

    int i, iter, conv, errFlag;
    int Nwant = es.nwant;
    int Nstep = es.nstep;
    int Asize = 2 * Nd;
    int mNwant = Nwant + 2;
    double res;

    size_t z_size = Nstep * Nstep * sizeof(cuDoubleComplex);

    /* Variables for lapack */
    lapack_int  n, lapack_info, ldz;
    n           = (lapack_int) Nstep;
    ldz         = n;

    cuDoubleComplex* U = lBuffer.dU;
    cuDoubleComplex* dz = lBuffer.dz;
    double* T0  = lBuffer.T0;
    double* T1  = lBuffer.T1;
    double *T2  = lBuffer.T2;
    double* LT0 = lBuffer.LT0;
    double* LT1 = lBuffer.LT1;
    cmpx *z = lBuffer.z;

    cuDoubleComplex one  = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
    /* Initial Decomposition */
    Lanczos_decomp_gpu(U, T0, T1, 0, cuHandles, fft_buffer, Lambdas_cuda, mtx_B, 
                             ls, Nx, Ny, Nz, Nd, Nwant, Nstep, flag_CompType, Profile);

    /* Begin Lanczos iteration */
    for(iter = 1; iter <= es.maxit; iter++)
    {
        memcpy(LT0, T0, Nstep * sizeof(double));
        memcpy(LT1, T1, (Nstep-1) * sizeof(double));

        /* Get the Ritz values T_d and Ritz vectors z*/
        /* Note that T_d will stored in descending order */
        lapack_info = LAPACKE_zpteqr(LAPACK_COL_MAJOR, 'I', n, LT0, LT1, z, ldz);
        assert(lapack_info == 0);

        cudaMemcpy(dz, z, z_size, cudaMemcpyHostToDevice);

        /* Check convergence, T_e will store the residules */
        conv = 0;
        for(i = 0; i < Nwant; i++)
        {
            res = T1[Nstep - 1] * cabs(z[(i + 1) * Nstep - 1]);

            if(res < es.tol)
                    conv++;
            else
                break;
        }

        /* Converged!! */
        if(conv == Nwant)
            break;

        cublasStatus = cublasZgemm(cuHandles.cublas_handle,CUBLAS_OP_N, CUBLAS_OP_N, Asize, mNwant, Nstep, &one, U, Asize, 
            dz, Nstep, &zero, ev, Asize);
        assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 
        
        errFlag = Lanczos_LockPurge(cuHandles, &lBuffer, ev,  mNwant-1, Nstep, Asize );
        assert( errFlag == 0 );

        checkCudaErrors(cudaMemcpy(U, ev, sizeof(cuDoubleComplex) * Asize * mNwant, cudaMemcpyDeviceToDevice));
        memcpy(T0, LT0, mNwant * sizeof(double));
        memcpy(T1, T2, (mNwant-1) * sizeof(double));

        checkCudaErrors(cudaMemcpy(U+mNwant*Asize, U+Nstep*Asize, sizeof(cuDoubleComplex) * Asize, cudaMemcpyDeviceToDevice)); 
        T1[mNwant-1]=T2[mNwant-1]*T1[Nstep-1];
 
        printf("\033[40;33m= = = = = = = = = = = = = = = LANCZOS Restart : %2d = = = = = = = = = = = = = = =\033[0m\n", iter);
        /* Restart */
        Lanczos_decomp_gpu(U, T0, T1, 1, cuHandles, fft_buffer, Lambdas_cuda, mtx_B, 
                                 ls, Nx, Ny, Nz, Nd, mNwant, Nstep, flag_CompType, Profile);
    }
    if(iter == es.maxit + 1)
        printf("\033[40;31mLANCZOS did not converge when restart numbers reached ES_MAXIT (%3d).\033[0m\n", es.maxit);
    
    for(i = 0; i < Nwant; i++)
        Freq_array[i] = sqrt(1.0 / LT0[i]);

    cublasZgemm(cuHandles.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, Asize, Nwant, Nstep, &one, U, Asize, dz, Nstep, &zero, ev, Asize);

    return iter;
}
