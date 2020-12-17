#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include <complex.h>
#include <vector>
#include "lapacke.h"
//#include "mkl.h"
#include "CG_Biiso.cuh"
#include "Lanczos_decomp_Biisotropic.cuh"
#include "Lanczos_LockPurge.cuh"
#include "FAME_Matrix_Vector_Production_Biisotropic_Ar.cuh"
#include "FAME_Matrix_Vector_Production_Biisotropic_Posdef.cuh"
#include "printDeviceArray.cuh"


static __global__ void Lanczos_init_kernel(cuDoubleComplex *U, const int size);

void quickSort (double* A,int* idx, int left, int right);
int partition(double* arr, int *idx, int left, int right);
void swap(double *arr, int *idx, int i, int j);

int Lanczos_Biisotropic
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
        double*          Freq_array, 
        cuDoubleComplex* ev,
        string flag_CompType,
        PROFILE* Profile)
        
{
    cout << "In Lanczos_Biisotropic" << endl;
	cublasHandle_t cublas_handle = cuHandles.cublas_handle;
	cublasStatus_t cublasStatus;

    int Nwant = es.nwant;
    int mNwant=Nwant+2;
    int Nstep = es.nstep;
    double ls_tol = ls.tol;
    double eig_tol = es.tol;
    int e_max_iter = es.maxit;
    int l_max_iter = ls.maxit;
    
    int conv, tmpIdx, errFlag, iter, i;
    cuDoubleComplex cublas_zcale, alpha_tmp, beta_tmp, Loss;
    double cublas_scale;
    int size = 4 * Nd;

    dim3 DimBlock( BLOCK_SIZE, 1, 1);
    dim3 DimGrid( (size-1)/BLOCK_SIZE +1, 1, 1);
 
    size_t memsize;
    size_t z_size = Nstep * Nstep * sizeof(cuDoubleComplex);
    int ewidx[9];

    lapack_int  n, lapack_info, ldz; 
    n = (lapack_int) Nstep;
    ldz = n;
    double vl=0,vu=10;
    lapack_int m,il=1,iu=1;
    double* ew    = (double*) calloc(Nstep, sizeof(double));
    lapack_int *isuppz    = (lapack_int*) calloc(2*Nstep , sizeof(lapack_int));
    int *number =(int*) calloc(Nstep, sizeof(int));
    

    cuDoubleComplex* dz  = lBuffer.dz;
    cuDoubleComplex *U   = lBuffer.dU;
    double *T0           = lBuffer.T0;
    double *T1           = lBuffer.T1;
    double *T2           = lBuffer.T2;
    double *LT0          = lBuffer.LT0;
    double *LT1          = lBuffer.LT1;
    cmpx *z              = lBuffer.z;

    cuDoubleComplex *w, *p, *r;
    memsize = size * sizeof(cuDoubleComplex);
    checkCudaErrors(cudaMalloc((void**)&w, memsize));
    checkCudaErrors(cudaMalloc((void**)&p, memsize));
	checkCudaErrors(cudaMalloc((void**)&r, memsize));
   
    cmpx *temp_z    = (cmpx*) calloc(Nstep*Nstep, sizeof(cmpx));

    cuDoubleComplex one  = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

    Lanczos_init_kernel<<<DimGrid, DimBlock>>>(U, size);
 
    //cout<<"j = "<< 0 <<"================================="<<endl;

    if(flag_CompType == "Simple")
    {
        FAME_Matrix_Vector_Production_Biisotropic_Ar(cuHandles, fft_buffer,
             U, mtx_B, Nx, Ny, Nz, Nd, 
            Pi_Qr, Pi_Pr,Pi_Qrs, Pi_Prs,
            D_k, D_ks, r, Profile);
    }
    else if(flag_CompType == "General")
    {
        FAME_Matrix_Vector_Production_Biisotropic_Ar(cuHandles, fft_buffer, 
             U, mtx_B, Nx, Ny, Nz, Nd, 
            Pi_Qr, Pi_Pr,Pi_Qrs, Pi_Prs,
            D_kx, D_ky, D_kz, r, Profile);
    }

    cublasStatus = cublasZdotc_v2(cublas_handle, size, U, 1, r, 1, &beta_tmp);
    assert( cublasStatus == CUBLAS_STATUS_SUCCESS );

    T1[0] = sqrt(beta_tmp.x);

    cublas_scale = 1.0 / T1[0];
    cublasStatus = cublasZdscal_v2( cublas_handle, size, &(cublas_scale), U, 1 );
    assert( cublasStatus == CUBLAS_STATUS_SUCCESS );

 
    FAME_Matrix_Vector_Production_Biisotropic_Posdef(cuHandles, 
                     U, mtx_B, Nx, Ny, Nz, Nd, Lambda_q_sqrt,
                    Pi_Qr, Pi_Pr,Pi_Qrs, Pi_Prs,
                    D_k, D_ks, w);

    cublasStatus = cublasZdotc_v2(cublas_handle, size, U, 1, w, 1, &alpha_tmp);

    checkCudaErrors(cudaMemcpy(p, r, memsize, cudaMemcpyDeviceToDevice));
    cublasStatus = cublasZdscal_v2(cublas_handle, size, &(cublas_scale), p, 1);
    assert( cublasStatus == CUBLAS_STATUS_SUCCESS );

    cublas_zcale = make_cuDoubleComplex(-alpha_tmp.x, 0.0);
    cublasStatus = cublasZaxpy_v2(cublas_handle, size, &cublas_zcale, p, 1, w, 1);
    assert( cublasStatus == CUBLAS_STATUS_SUCCESS );

    checkCudaErrors(cudaMemcpy(r, w, memsize, cudaMemcpyDeviceToDevice));

    /* orthogonalization */
    cublasStatus = cublasZdotc_v2(cublas_handle, size, U, 1, r, 1, &Loss );
    assert( cublasStatus == CUBLAS_STATUS_SUCCESS );


    cublas_zcale = make_cuDoubleComplex( -Loss.x, -Loss.y );
    cublasStatus = cublasZaxpy_v2(cublas_handle, size, &cublas_zcale, p, 1, r, 1);
    assert( cublasStatus == CUBLAS_STATUS_SUCCESS );  
    T0[0] = alpha_tmp.x + Loss.x;

    // Time start 
    int itercg;
    struct timespec start, end;
    clock_gettime (CLOCK_REALTIME, &start);

    if(flag_CompType == "Simple")
    {
        itercg = CG_Biiso(cuHandles, fft_buffer, mtx_B,ls_tol,l_max_iter, r, Nx, Ny, Nz, Nd,  D_k, D_ks, Pi_Qr,Pi_Pr, Pi_Qrs,Pi_Prs,  U+size, Profile);
    }
    else if(flag_CompType == "General")
    {
        itercg = CG_Biiso(cuHandles, fft_buffer, mtx_B,ls_tol,l_max_iter, r, Nx, Ny, Nz, Nd, D_kx, D_ky, D_kz,  Pi_Qr,Pi_Pr, Pi_Qrs,Pi_Prs,  U+size, Profile);
    }
    
    clock_gettime (CLOCK_REALTIME, &end);
	Profile->ls_iter[Profile->idx] += itercg;
    Profile->ls_time[Profile->idx] += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / BILLION;

   
    cublasStatus = cublasZdotc_v2(cublas_handle, size, U+size, 1, r, 1, &beta_tmp);
    assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 
    T1[1] = sqrt(beta_tmp.x);

    cublas_scale = 1.0 / T1[1];
    cublasStatus = cublasZdscal_v2( cublas_handle, size, &(cublas_scale), U+size, 1 );
    assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 

    /* Initial Decomposition */
    errFlag = Lanczos_decomp_Biisotropic(cuHandles, fft_buffer, U, mtx_B, Nx, Ny, Nz, Nd,  ls, Lambda_q_sqrt,
        Pi_Qr, Pi_Pr, Pi_Qrs, Pi_Prs, D_k, D_ks, D_kx, D_ky, D_kz, T0, T1, flag_CompType, 1, 2*mNwant, Profile);
    assert(errFlag == 0);
    //cout<<"Initial Lanczos decomposition completed!"<<endl;
    
    int restart_num = 0;

    for(iter = 1; iter <= e_max_iter; iter++)
    {
        errFlag = Lanczos_decomp_Biisotropic(cuHandles, fft_buffer, U, mtx_B, Nx, Ny, Nz, Nd, ls, Lambda_q_sqrt,
            Pi_Qr, Pi_Pr, Pi_Qrs, Pi_Prs, D_k, D_ks, D_kx, D_ky, D_kz, T0, T1, flag_CompType, 2*mNwant, Nstep, Profile);
        assert(errFlag == 0);

        //cout<<"Lanczos decomposition completed!"<<endl;
        memcpy(LT0, T0, Nstep * sizeof(double));
        memcpy(LT1, T1+1, (Nstep-1) * sizeof(double));  

        lapack_info = LAPACKE_zstegr(LAPACK_COL_MAJOR, 'V','A', n, LT0, LT1, vl, vu, il, iu, eig_tol, &m, ew, temp_z, ldz, isuppz);
        assert(lapack_info == 0);
 
        for(int i=0;i<Nstep;i++)
            number[i]=i;

        quickSort(ew,number, 0, Nstep-1);

        for(int i=0;i<Nstep;i++)
            memcpy(z+i*Nstep, temp_z+Nstep*number[i], Nstep* sizeof(cmpx));   

        memcpy(LT0, ew, Nstep* sizeof(double));             
        checkCudaErrors(cudaMemcpy(dz, z, z_size, cudaMemcpyHostToDevice));

        conv = 0;   

        for (int i=0; i<Nstep; i++)
        {
            tmpIdx = (i+1)*Nstep - 1;
            LT1[i] = cabs(T1[Nstep] * z[tmpIdx] );
            if ( LT0[i]>0 )
            {
                if ( LT1[i] < eig_tol )
                {
                    ewidx[conv]=i;
                    conv++;
                    if ( conv == Nwant )
                        break;              
                }
                else
                    break;
            }
            
        } 

        if ( conv >= Nwant )
            break;

        /* max iteration */
        if ( iter == e_max_iter )
            break;

        /* Implicit Restart: Lock and Purge */
        cublasStatus = cublasZgemm(cublas_handle,CUBLAS_OP_N, CUBLAS_OP_N,size, 2*mNwant, Nstep, &one, U, size, 
            dz, Nstep, &zero, ev, size);
        assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 

        errFlag = Lanczos_LockPurge(cuHandles, &lBuffer, ev, 2*mNwant-1, Nstep, size );
        assert( errFlag == 0 );
        //cout<<"Implicit Restart complete!"<<endl;   

        checkCudaErrors(cudaMemcpy(U, ev, sizeof(cuDoubleComplex)*size*2*mNwant, cudaMemcpyDeviceToDevice)); 


        memcpy(T0, LT0, 2*mNwant * sizeof(double));
        memcpy(T1+1, T2, (2*mNwant-1) * sizeof(double));

        checkCudaErrors(cudaMemcpy(U+2*mNwant*size, U+Nstep*size, sizeof(cuDoubleComplex)*size, cudaMemcpyDeviceToDevice)); 
        T1[2*mNwant]=T2[2*mNwant-1]*T1[Nstep];

        restart_num ++;
        
    }

    for(i = 0; i < Nwant; i++) 
    { 
        Freq_array[i] = 1.0 / LT0[ewidx[i]]; 
        memcpy(temp_z+i*Nstep, z+ewidx[i]*Nstep, Nstep * sizeof(cmpx));  
    }

    Profile->es_iter[Profile->idx] = Nstep + (Nstep-2*mNwant) * restart_num;

    checkCudaErrors(cudaMemcpy(dz, temp_z, Nwant*Nstep*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));


    cublasStatus = cublasZgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, size, Nwant, Nstep, &one, U, size,
         dz, Nstep, &zero, ev, size);
    assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 

    free(temp_z); free(ew);free(number);free(isuppz);
    cudaFree(w);cudaFree(r);cudaFree(p);
      
    return iter;

}

void quickSort ( double* A, int* idx, int left, int right ) 
{ 
    int partitionIndex;
    if (left < right) 
    {
        partitionIndex = partition(A, idx, left, right);
        quickSort(A, idx,left, partitionIndex-1);
        quickSort(A, idx,partitionIndex+1, right);
    }
}
 
int partition ( double* arr, int *idx, int left, int right )
{     
    int pivot = left,                      
        index = pivot + 1;
    for (int i = index; i <= right; i++) {
        if (fabs(arr[i]) > fabs(arr[pivot])) {
            swap(arr, idx, i, index);
            index++;
        }       
    }
    swap(arr, idx, pivot, index - 1);
    return index-1;
}
 
void swap ( double *arr, int *idx, int i, int j ) 
{
    double temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
    int tep=idx[i];
    idx[i] = idx[j];
    idx[j] = tep;
}


static __global__ void Lanczos_init_kernel(cuDoubleComplex *U, const int size)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if ( idx < size - 1 )
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
