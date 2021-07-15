#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include <complex.h>
#include <vector>
#include "lapacke.h"
#include "quickSort.cuh"
#include "CG_FdualF.cuh"
#include "Lanczos_decomp_Bianisotropic.cuh"
#include "Lanczos_LockPurge.cuh"
#include "FAME_Matrix_Vector_Production_Bianisotropic_FdualF.cuh"
#include "FAME_Matrix_Vector_Production_Bianisotropic_i_invSigma.cuh"
#include "printDeviceArray.cuh"


static __global__ void Lanczos_init_kernel(cmpxGPU *U, const int size);


int Lanczos_Bianisotropic
( 	CULIB_HANDLES cuHandles, 
    FFT_BUFFER    fft_buffer,
    LANCZOS_BUFFER lBuffer,
		MTX_B mtx_B,
        int Nx,
        int Ny,
        int Nz,
        int Nd,
        ES  es,
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
        realGPU* Freq_array, 
        cmpxGPU* ev,
        string flag_CompType,
        PROFILE* Profile)
        
{

    cout << "In Lanczos_Bi-anisotropic" << endl;

	cublasHandle_t cublas_handle = cuHandles.cublas_handle;
	cublasStatus_t cublasStatus;

    int Nwant = es.nwant;
    int mNwant = Nwant+2;
    int Nstep = es.nstep;
    realGPU ls_tol = ls.tol;
    realGPU eig_tol = es.tol;
    int e_max_iter = es.maxit;
    int l_max_iter = ls.maxit;
    
    int conv, tmpIdx, errFlag, iter, i;
    cmpxGPU cublas_zcale, alpha_tmp, beta_tmp, Loss;
    realGPU cublas_scale;
    int size = 32 * Nd;

    dim3 DimBlock( BLOCK_SIZE, 1, 1);
    dim3 DimGrid( (size-1)/BLOCK_SIZE +1, 1, 1);
 
    size_t memsize;
    size_t z_size = Nstep * Nstep * sizeof(cmpxGPU);
    int *ewidx = (int *)calloc(Nwant, sizeof(int));

    lapack_int  n, lapack_info, ldz; 
    n = (lapack_int) Nstep;
    ldz = n;
    realGPU vl=0, vu=10;
    lapack_int m, il=1, iu=1;
    realGPU* ew = (realGPU*) calloc(Nstep, sizeof(realGPU));
    lapack_int *isuppz = (lapack_int*) calloc(2*Nstep , sizeof(lapack_int));
    int *number = (int*) calloc(Nstep, sizeof(int));

    cmpxGPU* dz  = lBuffer.dz;
    cmpxGPU *U   = lBuffer.dU;
    realGPU *T0  = lBuffer.T0;
    realGPU *T1  = lBuffer.T1;
    realGPU *T2  = lBuffer.T2;
    realGPU *LT0 = lBuffer.LT0;
    realGPU *LT1 = lBuffer.LT1;
    cmpxCPU *z   = lBuffer.z;

    cmpxGPU *w, *p, *r;
    memsize = size * sizeof(cmpxGPU);
    checkCudaErrors(cudaMalloc((void**)&w, memsize));
    checkCudaErrors(cudaMalloc((void**)&p, memsize));
	checkCudaErrors(cudaMalloc((void**)&r, memsize));
   
    cmpxCPU *temp_z = (cmpxCPU*) calloc(Nstep*Nstep, sizeof(cmpxCPU));

    cmpxGPU one  = make_cucmpx(1.0, 0.0);
    cmpxGPU zero = make_cucmpx(0.0, 0.0);

    Lanczos_init_kernel<<<DimGrid, DimBlock>>>(U, size);
    cudaDeviceSynchronize();

    if(flag_CompType == "Simple")
    {
        FAME_Matrix_Vector_Production_Bianisotropic_FdualF(cuHandles, fft_buffer, mtx_B, Nx, Ny, Nz, Nd, 
            Pi_Qr, Pi_Pr, Pi_Qrs, Pi_Prs, Pi_Qr_110, Pi_Pr_110, Pi_Qrs_110, Pi_Prs_110,
            Pi_Qr_101, Pi_Pr_101, Pi_Qrs_101, Pi_Prs_101, Pi_Qr_011, Pi_Pr_011, Pi_Qrs_011, Pi_Prs_011,
            D_k, D_ks, U, r);
    }
    else if(flag_CompType == "General")
    {
        FAME_Matrix_Vector_Production_Bianisotropic_FdualF(cuHandles, fft_buffer, mtx_B, Nx, Ny, Nz, Nd, 
            Pi_Qr, Pi_Pr, Pi_Qrs, Pi_Prs, Pi_Qr_110, Pi_Pr_110, Pi_Qrs_110, Pi_Prs_110,
            Pi_Qr_101, Pi_Pr_101, Pi_Qrs_101, Pi_Prs_101, Pi_Qr_011, Pi_Pr_011, Pi_Qrs_011, Pi_Prs_011,
            D_kx, D_ky, D_kz, U, r);
    }

    cublasStatus = FAME_cublas_dot(cublas_handle, size, U, 1, r, 1, &beta_tmp);
    assert( cublasStatus == CUBLAS_STATUS_SUCCESS );

    T1[0] = sqrt(beta_tmp.x);

    cublas_scale = 1.0 / T1[0];
    cublasStatus = FAME_cublas_dscal( cublas_handle, size, &(cublas_scale), U, 1 );
    assert( cublasStatus == CUBLAS_STATUS_SUCCESS );

    FAME_Matrix_Vector_Production_Bianisotropic_i_invSigma(cuHandles, Nd, Lambda_q_sqrt, U, w);

    cublasStatus = FAME_cublas_dot(cublas_handle, size, U, 1, w, 1, &alpha_tmp);

    checkCudaErrors(cudaMemcpy(p, r, memsize, cudaMemcpyDeviceToDevice));
    cublasStatus = FAME_cublas_dscal(cublas_handle, size, &(cublas_scale), p, 1);
    assert( cublasStatus == CUBLAS_STATUS_SUCCESS );

    cublas_zcale = make_cucmpx(-alpha_tmp.x, 0.0);
    cublasStatus = FAME_cublas_axpy(cublas_handle, size, &cublas_zcale, p, 1, w, 1);
    assert( cublasStatus == CUBLAS_STATUS_SUCCESS );

    checkCudaErrors(cudaMemcpy(r, w, memsize, cudaMemcpyDeviceToDevice));

    /* orthogonalization */
    cublasStatus = FAME_cublas_dot(cublas_handle, size, U, 1, r, 1, &Loss );
    assert( cublasStatus == CUBLAS_STATUS_SUCCESS );


    cublas_zcale = make_cucmpx( -Loss.x, -Loss.y );
    cublasStatus = FAME_cublas_axpy(cublas_handle, size, &cublas_zcale, p, 1, r, 1);
    assert( cublasStatus == CUBLAS_STATUS_SUCCESS );  
    T0[0] = alpha_tmp.x + Loss.x;

    // Time start 
    int itercg;
    struct timespec start, end;
    clock_gettime (CLOCK_REALTIME, &start);
    if(flag_CompType == "Simple")
    {
        itercg = CG_FdualF(r, U+size, cuHandles, fft_buffer, mtx_B, Nx, Ny, Nz, Nd, l_max_iter, ls_tol, Pi_Qr, Pi_Pr, Pi_Qrs, Pi_Prs, 
            Pi_Qr_110, Pi_Pr_110, Pi_Qrs_110, Pi_Prs_110, Pi_Qr_101, Pi_Pr_101, Pi_Qrs_101, Pi_Prs_101, Pi_Qr_011, Pi_Pr_011, Pi_Qrs_011, Pi_Prs_011,
            D_k, D_ks);
    }
    else if(flag_CompType == "General")
    {
        itercg = CG_FdualF(r, U+size, cuHandles, fft_buffer, mtx_B, Nx, Ny, Nz, Nd, l_max_iter, ls_tol, Pi_Qr, Pi_Pr, Pi_Qrs, Pi_Prs, 
            Pi_Qr_110, Pi_Pr_110, Pi_Qrs_110, Pi_Prs_110, Pi_Qr_101, Pi_Pr_101, Pi_Qrs_101, Pi_Prs_101, Pi_Qr_011, Pi_Pr_011, Pi_Qrs_011, Pi_Prs_011,
            D_kx, D_ky, D_kz);    
    }
    clock_gettime (CLOCK_REALTIME, &end);
	Profile->ls_iter[Profile->idx] += itercg;
    Profile->ls_time[Profile->idx] += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / BILLION;

   
    cublasStatus = FAME_cublas_dot(cublas_handle, size, U+size, 1, r, 1, &beta_tmp);
    assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 
    T1[1] = sqrt(beta_tmp.x);

    cublas_scale = 1.0 / T1[1];
    cublasStatus = FAME_cublas_dscal( cublas_handle, size, &(cublas_scale), U+size, 1 );
    assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 

    /* Initial Decomposition */
    errFlag = Lanczos_decomp_Bianisotropic(cuHandles, fft_buffer, U, mtx_B, Nx, Ny, Nz, Nd, ls, Lambda_q_sqrt,
        Pi_Qr, Pi_Pr, Pi_Qrs, Pi_Prs, Pi_Qr_110, Pi_Pr_110, Pi_Qrs_110, Pi_Prs_110, Pi_Qr_101, Pi_Pr_101, Pi_Qrs_101, Pi_Prs_101, Pi_Qr_011, Pi_Pr_011, Pi_Qrs_011, Pi_Prs_011, 
        D_k, D_ks, D_kx, D_ky, D_kz, T0, T1, flag_CompType, 1, 2*mNwant, Profile);
    assert(errFlag == 0);
    //cout<<"Initial Lanczos decomposition completed!"<<endl;
    
    int restart_num = 0;

    for(iter = 1; iter <= e_max_iter; iter++)
    {
        errFlag = Lanczos_decomp_Bianisotropic(cuHandles, fft_buffer, U, mtx_B, Nx, Ny, Nz, Nd, ls, Lambda_q_sqrt,
            Pi_Qr, Pi_Pr, Pi_Qrs, Pi_Prs, Pi_Qr_110, Pi_Pr_110, Pi_Qrs_110, Pi_Prs_110, Pi_Qr_101, Pi_Pr_101, Pi_Qrs_101, Pi_Prs_101, Pi_Qr_011, Pi_Pr_011, Pi_Qrs_011, Pi_Prs_011, 
            D_k, D_ks, D_kx, D_ky, D_kz, T0, T1, flag_CompType, 2*mNwant, Nstep, Profile);
        assert(errFlag == 0);

        //cout<<"Lanczos decomposition completed!"<<endl;
        memcpy(LT0, T0, Nstep * sizeof(realGPU));
        memcpy(LT1, T1+1, (Nstep-1) * sizeof(realGPU));  

        lapack_info = FAME_lapacke_stegr(LAPACK_COL_MAJOR, 'V','A', n, LT0, LT1, vl, vu, il, iu, eig_tol, &m, ew, temp_z, ldz, isuppz);
        assert(lapack_info == 0);
 
        for(int i = 0; i < Nstep; i++)
        {
            number[i] = i;
        }

        quickSort(ew, number, 0, Nstep-1);

        for(int i = 0; i < Nstep; i++)
        {
            memcpy(z + i * Nstep, temp_z + Nstep * number[i], Nstep * sizeof(cmpxCPU));  
        } 

        memcpy(LT0, ew, Nstep * sizeof(realGPU));             
        checkCudaErrors(cudaMemcpy(dz, z, z_size, cudaMemcpyHostToDevice));

        conv = 0;   

        for (int i = 0; i < Nstep; i++)
        {
            tmpIdx = (i + 1) * Nstep - 1;
            
            LT1[i] = cabs(T1[Nstep] * z[tmpIdx] );
            if ( LT0[i] > 0 )
            {
                if ( LT1[i] < eig_tol )
                {
                    ewidx[conv] = i;
                    conv ++;
                    if ( conv == Nwant )
                        break;              
                }
                else
                    break;
            }
            
        }

        if ( conv >= Nwant )
        {
            break;
        }

        /* max iteration */
        if ( iter == e_max_iter )
        {
            break;
        }

        /* Implicit Restart: Lock and Purge */
        cublasStatus = FAME_cublas_gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, size, 2*mNwant, Nstep, &one, U, size, 
            dz, Nstep, &zero, ev, size);
        assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 

        errFlag = Lanczos_LockPurge(cuHandles, &lBuffer, ev, 2*mNwant-1, Nstep, size );
        assert( errFlag == 0 );

        checkCudaErrors(cudaMemcpy(U, ev, sizeof(cmpxGPU)*size*2*mNwant, cudaMemcpyDeviceToDevice)); 


        memcpy(T0, LT0, 2*mNwant * sizeof(realGPU));
        memcpy(T1+1, T2, (2*mNwant-1) * sizeof(realGPU));

        checkCudaErrors(cudaMemcpy(U+2*mNwant*size, U+Nstep*size, sizeof(cmpxGPU)*size, cudaMemcpyDeviceToDevice)); 
        T1[2*mNwant]=T2[2*mNwant-1]*T1[Nstep];

        restart_num ++;

        printf("\033[40;33m= = = = = = = = = = = = = = = LANCZOS Restart : %2d = = = = = = = = = = = = = = =\033[0m\n", iter);
        
    }

    for(i = 0; i < Nwant; i++) 
    { 
        Freq_array[i] = 1.0 / LT0[ewidx[i]]; 
        // printf("%f\n", Freq_array[i]);
        memcpy(temp_z + i * Nstep, z + ewidx[i] * Nstep, Nstep * sizeof(cmpxCPU));  
    }
    Profile->es_iter[Profile->idx] = Nstep + (Nstep - 2 * mNwant) * restart_num;
    checkCudaErrors(cudaMemcpy(dz, temp_z, Nwant * Nstep * sizeof(cmpxGPU), cudaMemcpyHostToDevice));

    cublasStatus = FAME_cublas_gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, size, Nwant, Nstep, &one, U, size,
         dz, Nstep, &zero, ev, size);
    assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 

    free(ewidx);
    free(temp_z);free(ew);free(number);free(isuppz);
    cudaFree(w);cudaFree(r);cudaFree(p);

    /*
    cmpxGPU *tmp1, *tmp2;
    cublasStatus_t cublasErr;
    realGPU res;
    cudaMalloc((void**)&tmp1, size * sizeof(cmpxGPU));
    cudaMalloc((void**)&tmp2, size * sizeof(cmpxGPU));
    for(i = 0; i < Nwant; i++)
    {
        FAME_Matrix_Vector_Production_Bianisotropic_FdualF(cuHandles, fft_buffer, mtx_B, Nx, Ny, Nz, Nd, 
            Pi_Qr, Pi_Pr, Pi_Qrs, Pi_Prs, Pi_Qr_110, Pi_Pr_110, Pi_Qrs_110, Pi_Prs_110,
            Pi_Qr_101, Pi_Pr_101, Pi_Qrs_101, Pi_Prs_101, Pi_Qr_011, Pi_Pr_011, Pi_Qrs_011, Pi_Prs_011,
            D_kx, D_ky, D_kz, ev + i * size, tmp1);
        FAME_Matrix_Vector_Production_Bianisotropic_i_invSigma(cuHandles, Nd, Lambda_q_sqrt, ev + i * size, tmp2);
        cublas_zcale = make_cucmpx(-Freq_array[i], 0);
        cublasErr = FAME_cublas_axpy(cublas_handle, size, &cublas_zcale, tmp2, 1, tmp1, 1);
        assert( cublasErr == CUBLAS_STATUS_SUCCESS );
        FAME_cublas_nrm2(cublas_handle, size, tmp1, 1, &res);
        printf("res: %e.\n", res);
    }
    cudaFree(tmp1); cudaFree(tmp2);
    getchar();
    */
      
    return iter;

}


static __global__ void Lanczos_init_kernel(cmpxGPU *U, const int size)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if ( idx < size )
    { 
        U[idx].x = 1.0;
        U[idx].y = 0.0;
    }

    if ( idx == 0 )
    {
        U[idx].x = 1.0;
        U[idx].y = 0.0;
    }

}
