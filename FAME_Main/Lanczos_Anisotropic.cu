#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include "Lanczos_decomp_Anisotropic.cuh"
#include "Lanczos_LockPurge.cuh"
#include "cuda_profiler_api.h"
#include <lapacke.h>
#include "FAME_Matrix_Vector_Production_Anisotropic_invAr.cuh"
#include "printDeviceArray.cuh"
#include "FAME_Matrix_Vector_Production_Anisotropic_QBQ.cuh"


//static __global__ void dot_product(cmpxGPU* vec_y, realCPU* array, int size);
//static __global__ void dot_product(cmpxGPU* vec_y, cmpxGPU* vec_x, realGPU* Lambda_q_sqrt, int size);

int Lanczos_Anisotropic( 
    realCPU*         Freq_array, 
    cmpxGPU*         ev,
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
    cout << "In Lanczos_Anisotropic" << endl;
    cublasStatus_t cublasStatus;

    int i, iter, conv, errFlag;
    int Nwant = es.nwant;
    int Nstep = es.nstep;
    int Asize = 8 * Nd;
    int mNwant = Nwant + 2;
    realGPU res;

    size_t z_size = Nstep * Nstep * sizeof(cmpxGPU);

    /* Variables for lapack */
    lapack_int  n, lapack_info, ldz;
    n  = (lapack_int) Nstep;
    ldz = n;

    cmpxGPU* U   = lBuffer.dU;
    cmpxGPU* dz  = lBuffer.dz;
    realGPU* T0  = lBuffer.T0;
    realGPU* T1  = lBuffer.T1;
    realGPU *T2  = lBuffer.T2;
    realGPU* LT0 = lBuffer.LT0;
    realGPU* LT1 = lBuffer.LT1;
    cmpxCPU *z   = lBuffer.z;

    cmpxGPU one  = make_cucmpx(1.0, 0.0);
    cmpxGPU zero = make_cucmpx(0.0, 0.0);
    /* Initial Decomposition */
    Lanczos_decomp_Anisotropic(U, T0, T1, 0, cuHandles, fft_buffer, Lambdas_cuda, mtx_B, 
                             ls, Nx, Ny, Nz, Nd, Nwant, Nstep, flag_CompType, Profile);

    /* Begin Lanczos iteration */
    for(iter = 1; iter <= es.maxit; iter++)
    {
        memcpy(LT0, T0, Nstep * sizeof(realGPU));
        memcpy(LT1, T1, (Nstep-1) * sizeof(realGPU));

        /* Get the Ritz values T_d and Ritz vectors z*/
        /* Note that T_d will stored in descending order */
        lapack_info = FAME_lapacke_pteqr(LAPACK_COL_MAJOR, 'I', n, LT0, LT1, z, ldz);
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


        cublasStatus = FAME_cublas_gemm(cuHandles.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, Asize, mNwant, Nstep, &one, U, Asize, 
            dz, Nstep, &zero, ev, Asize);
        assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 

        errFlag = Lanczos_LockPurge(cuHandles, &lBuffer, ev,  mNwant-1, Nstep, Asize );
        assert( errFlag == 0 );

        checkCudaErrors(cudaMemcpy(U, ev, sizeof(cmpxGPU) * Asize * mNwant, cudaMemcpyDeviceToDevice));
        memcpy(T0, LT0, mNwant * sizeof(realGPU));
        memcpy(T1, T2, (mNwant-1) * sizeof(realGPU));

        checkCudaErrors(cudaMemcpy(U+mNwant*Asize, U+Nstep*Asize, sizeof(cmpxGPU) * Asize, cudaMemcpyDeviceToDevice)); 
        T1[mNwant-1] = T2[mNwant-1] * T1[Nstep-1];
 
        printf("\033[40;33m= = = = = = = = = = = = = = = LANCZOS Restart : %2d = = = = = = = = = = = = = = =\033[0m\n", iter);
        /* Restart */
        Lanczos_decomp_Anisotropic(U, T0, T1, 1, cuHandles, fft_buffer, Lambdas_cuda, mtx_B, 
                                 ls, Nx, Ny, Nz, Nd, mNwant, Nstep, flag_CompType, Profile);
    }
    if(iter == es.maxit + 1)
        printf("\033[40;31mLANCZOS did not converge when restart numbers reached ES_MAXIT (%3d).\033[0m\n", es.maxit);
    
    for(i = 0; i < Nwant; i++)
    {
        Freq_array[i] = sqrt(1.0 / LT0[i]);
    }

    cublasStatus = FAME_cublas_gemm(cuHandles.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, Asize, Nwant, Nstep, &one, U, Asize, dz, Nstep, &zero, ev, Asize);
    assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 

 
  /////Lanczos residual 
  /*  dim3 DimBlock(BLOCK_SIZE, 1, 1);
    dim3 DimGrid((Asize - 1) / BLOCK_SIZE + 1, 1, 1);
    cmpxGPU* tmp = cuHandles.Nd2_temp1;
    cmpxGPU* tmp2 = cuHandles.Nd2_temp2;
    
    int idx = 0;
    cmpxCPU *res_inf = (cmpxCPU*) malloc(Asize * sizeof(cmpxCPU));
    for (int j=0;j<Nwant;j++)
    {
        dot_product<<<DimGrid, DimBlock>>>(tmp, ev+j*Asize, Lambdas_cuda.Lambda_q_sqrt, Asize);
       if(flag_CompType == "Simple")
	FAME_Matrix_Vector_Production_Isotropic_QBQ(tmp2,tmp, cuHandles, fft_buffer, mtx_B,
                                     Lambdas_cuda.dD_k,  Lambdas_cuda.dD_ks,  Lambdas_cuda.dPi_Qr,  Lambdas_cuda.dPi_Qrs, Nx, Ny, Nz, Nd, Profile);
	else if(flag_CompType == "General") 
        FAME_Matrix_Vector_Production_Isotropic_QBQ(tmp2, tmp, cuHandles, fft_buffer, mtx_B, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, 
              Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Qr, Lambdas_cuda.dPi_Qrs, Nx, Ny, Nz, Nd,Profile);
              
        dot_product<<<DimGrid, DimBlock>>>(tmp2, Lambdas_cuda.Lambda_q_sqrt, Asize);

        cmpxGPU cublas_zcale = make_cucmpx(-1/LT0[j], 0.0);
        FAME_cublas_axpy(cuHandles.cublas_handle, Asize, &cublas_zcale, ev+j*Asize, 1, tmp2, 1);    
        FAME_cublas_nrm2(cuHandles.cublas_handle, Asize, tmp2, 1, &res);
        FAME_cublas_amax(cuHandles.cublas_handle, Asize, tmp2, 1, &idx);
        
        checkCudaErrors(cudaMemcpy(res_inf, tmp2, Asize*sizeof(cmpxGPU), cudaMemcpyDeviceToHost));

        printf("\033[40;31m idx %2d, Lanczos residual = %e\033[0m, residual_inf = %e.\033[0m \n", j, res, cabs(res_inf[idx-1]));
        //cout<<"res "<<res<<endl;    
    }
    cudaFree(tmp2);*/

    return iter;
}

/*static __global__ void dot_product(cmpxGPU* vec_y, realCPU* array, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
    {
        vec_y[idx].x = vec_y[idx].x * array[idx];
		vec_y[idx].y = vec_y[idx].y * array[idx];
    }

}
static __global__ void dot_product(cmpxGPU* vec_y, cmpxGPU* vec_x, realGPU* Lambda_q_sqrt, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
    {
        vec_y[idx].x = vec_x[idx].x * Lambda_q_sqrt[idx];
		    vec_y[idx].y = vec_x[idx].y * Lambda_q_sqrt[idx];
    }

}*/
