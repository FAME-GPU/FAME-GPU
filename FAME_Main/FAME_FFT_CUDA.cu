#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include "printDeviceArray.cuh"

#define NN 16
#define N1 1

static __global__ void D_k_product(cuDoubleComplex* vec_y, cuDoubleComplex* dD_k, cuDoubleComplex* vec_x, int N);
static __global__ void D_ks_product(cuDoubleComplex* vec_y, cuDoubleComplex* dD_k, cuDoubleComplex* vec_x, int N);

static __global__ void ConjMulti_Conj_transpose_1_1616(cuDoubleComplex *odata, cuDoubleComplex *idata, int n1, int n2, int n3, cuDoubleComplex* mtx);
static __global__ void Dan_Multi_transpose_1_1616(cuDoubleComplex *odata, cuDoubleComplex *idata, int n1, int n2, int n3, cuDoubleComplex* mtx);
static __global__ void DanMulti_conj(cuDoubleComplex* out, cuDoubleComplex *d_Dan, cuDoubleComplex *p, const int n);
static __global__ void Dan_Multi_trans(cuDoubleComplex* out, cuDoubleComplex* in, cuDoubleComplex* mtx, int n, int n1, int n2, int n3, int Gx, int Gz, double one_over_Gx, double one_over_Gz, int k2);
static __global__ void Dan_Multi_Complex_scale(cuDoubleComplex* out, cuDoubleComplex *d_Dan, cuDoubleComplex *p, int n);
static __global__ void Conj_Dan_trans(cuDoubleComplex* out, cuDoubleComplex* in, cuDoubleComplex* mtx, int n, int n1, int n2, int n3, int Gx, int Gz, double one_over_Gx, double one_over_Gz, int k2);

////////////=========================== Create FFT (simple) function for Biiso (cuda)===========================//////////////////
int FFT_CUDA(cuDoubleComplex* vec_y, cuDoubleComplex* vec_x, cuDoubleComplex* dD_ks, FFT_BUFFER fftBuffer, CULIB_HANDLES cuHandles, int Nx, int Ny, int Nz)
{

    int N = Nx * Ny * Nz;
    int N2 = N * 2;
    cufftResult cufftRe;
    cublasStatus_t cublasStatus;
    double alpha = 1 / sqrt((double)N);

    dim3 DimBlock(BLOCK_SIZE, 1, 1);
    dim3 DimGrid((N*3 - 1) / BLOCK_SIZE + 1, 1, 1);

    cuDoubleComplex* temp = fftBuffer.d_A;

    D_ks_product<<<DimGrid, DimBlock>>>(temp, dD_ks, vec_x, N);

    cufftRe = cufftExecZ2Z(cuHandles.cufft_plan, temp,    vec_y,    CUFFT_FORWARD);
    assert( cufftRe == CUFFT_SUCCESS );
    cufftRe = cufftExecZ2Z(cuHandles.cufft_plan, temp+N,  vec_y+N,  CUFFT_FORWARD);
    assert( cufftRe == CUFFT_SUCCESS );
    cufftRe = cufftExecZ2Z(cuHandles.cufft_plan, temp+N2, vec_y+N2, CUFFT_FORWARD);
    assert( cufftRe == CUFFT_SUCCESS );

    cublasStatus = cublasZdscal_v2(cuHandles.cublas_handle, N * 3, &alpha, vec_y, 1);
    assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 

    return 0;
}

int IFFT_CUDA(cuDoubleComplex* vec_y, cuDoubleComplex* vec_x, cuDoubleComplex* dD_k, FFT_BUFFER fftBuffer, CULIB_HANDLES cuHandles, int Nx, int Ny, int Nz)
{

    int N = Nx * Ny * Nz;
    int N2 = N * 2;
    cufftResult cufftRe;
    cublasStatus_t cublasStatus;
    dim3 DimBlock(BLOCK_SIZE, 1, 1);
    dim3 DimGrid((N*3 - 1) / BLOCK_SIZE + 1, 1, 1);
    
    // double alpha = 1.0 / (double)N;
    
    double alpha = 1.0 / sqrt((double) N);

    cuDoubleComplex* temp = fftBuffer.d_A;

    cufftRe = cufftExecZ2Z(cuHandles.cufft_plan, vec_x,    temp,    CUFFT_INVERSE);
    assert( cufftRe == CUFFT_SUCCESS );

    cufftRe = cufftExecZ2Z(cuHandles.cufft_plan, vec_x+N,  temp+N,  CUFFT_INVERSE);
    assert( cufftRe == CUFFT_SUCCESS );
    cufftRe = cufftExecZ2Z(cuHandles.cufft_plan, vec_x+N2, temp+N2, CUFFT_INVERSE);
    assert( cufftRe == CUFFT_SUCCESS );

    D_k_product<<<DimGrid, DimBlock>>>(vec_y, dD_k, temp, N);

    cublasStatus =cublasZdscal_v2(cuHandles.cublas_handle, N * 3, &alpha, vec_y, 1);
    assert( cublasStatus == CUBLAS_STATUS_SUCCESS ); 
    return 0;
}

/* main work function */
int spMV_fastT_gpu( 
    cuDoubleComplex *out,
    cuDoubleComplex *p,
    CULIB_HANDLES cuHandles,
    FFT_BUFFER *fftBuffer, 
    cuDoubleComplex *mtx_D_kx,   // D_kx
    cuDoubleComplex *mtx_D_jx,   // D_ky
    cuDoubleComplex *mtx_D_jell, // D_kz 
    const int n1, 
    const int n2, 
    const int n3, 
    const int flag)
{

    int n = n1*n2*n3;
    cufftResult    cufftErr;
    dim3 DimBlock(BLOCK_SIZE, 1, 1);
    dim3 DimGrid( (n-1)/BLOCK_SIZE+1, 1, 1);

    // for Dan_Multi_trans
    int k1, k2, Gx, Gz;
    double db_n2 = (double)(n2);
    Gx = (n1 + BLOCK_DIM_TR-1)/BLOCK_DIM_TR;
    Gz = (n3 + BLOCK_DIM_TR-1)/BLOCK_DIM_TR;
    int max_k1 = (int) floor( sqrt(db_n2));
    for ( k1 = max_k1 ; 1 <= k1 ; k1--){
        k2 = (int) ceil( db_n2/((double)k1));
        if ( 1>= (k1*k2 - n2))
            break;
    }
    double one_over_Gx = 1.0/((double)Gx);
    double one_over_Gz = 1.0/((double)Gz);
    dim3 DimBlock_dantr(BLOCK_DIM_TR, BLOCK_DIM_TR, 1);
    dim3 DimGrid_dantr(k2*Gz, k1*Gx, 1);

    dim3 DimBlock_1616(NN, NN, N1);
//  dim3 DimGrid_1616((n1-1)/NN +1,(n2-1)/NN +1,(n3-1)/N1 +1);
    dim3 DimGrid_1616((n3-1)/NN +1,(n2-1)/NN +1,(n1-1)/N1 +1);
    switch (flag)
    {
        /* (T) p */
        case 1:
        {
            cufftErr = cufftExecZ2Z(cuHandles.cufft_plan_1d_z, p, p, CUFFT_INVERSE );
            assert(cufftErr == CUFFT_SUCCESS);

            Dan_Multi_transpose_1_1616<<<DimGrid_1616, DimBlock_1616>>>(fftBuffer->d_A, p, n1, n2, n3, mtx_D_jell);
            cudaDeviceSynchronize();

            cufftErr = cufftExecZ2Z(cuHandles.cufft_plan_1d_y, fftBuffer->d_A, fftBuffer->d_A, CUFFT_INVERSE );
            assert(cufftErr == CUFFT_SUCCESS);

            Dan_Multi_trans<<<DimGrid_dantr , DimBlock_dantr>>>(p, fftBuffer->d_A, mtx_D_jx, n, n1, n2, n3, Gx, Gz, one_over_Gx, one_over_Gz, k2);
            cudaDeviceSynchronize();

            cufftErr = cufftExecZ2Z(cuHandles.cufft_plan_1d_x, p, fftBuffer->d_A, CUFFT_INVERSE);
            assert(cufftErr == CUFFT_SUCCESS);

            Dan_Multi_Complex_scale<<<DimGrid, DimBlock>>>(p, mtx_D_kx, fftBuffer->d_A, n);
            cudaDeviceSynchronize();

            cublasZcopy_v2(cuHandles.cublas_handle, n, p, 1, out, 1);
            break;
        }
            /* (T*) p */
        case -1:
        {

            DanMulti_conj<<<DimGrid, DimBlock>>>(fftBuffer->d_A, mtx_D_kx, p, n);
            cudaDeviceSynchronize();

            cufftErr = cufftExecZ2Z(cuHandles.cufft_plan_1d_x, fftBuffer->d_A, fftBuffer->d_A, CUFFT_FORWARD );
            assert(cufftErr == CUFFT_SUCCESS);

            Conj_Dan_trans<<<DimGrid_dantr, DimBlock_dantr>>>(p, fftBuffer->d_A, mtx_D_jx, n, n1, n2, n3, Gx, Gz, one_over_Gx, one_over_Gz, k2);
            cudaDeviceSynchronize();

            cufftErr = cufftExecZ2Z(cuHandles.cufft_plan_1d_y, p, fftBuffer->d_A, CUFFT_FORWARD );
            assert(cufftErr == CUFFT_SUCCESS);

            ConjMulti_Conj_transpose_1_1616<<<DimGrid_1616, DimBlock_1616>>>(p, fftBuffer->d_A, n1, n2, n3, mtx_D_jell);
            cudaDeviceSynchronize();

            cufftErr = cufftExecZ2Z(cuHandles.cufft_plan_1d_z, p, p, CUFFT_FORWARD );
            assert(cufftErr == CUFFT_SUCCESS);

            cublasZcopy_v2(cuHandles.cublas_handle, n, p, 1, out, 1);
            break;
        }
        default:
        {
            printf("Error input of flag \n");
            return -1;
        }

    }
    return 0;
}

static __global__ void D_k_product(cuDoubleComplex* vec_y, cuDoubleComplex* dD_k, cuDoubleComplex* vec_x, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N * 3)
    {
        vec_y[idx].x = dD_k[idx % N].x * vec_x[idx].x - dD_k[idx % N].y * vec_x[idx].y;
        vec_y[idx].y = dD_k[idx % N].x * vec_x[idx].y + dD_k[idx % N].y * vec_x[idx].x; 
    }
}

static __global__ void D_ks_product(cuDoubleComplex* vec_y, cuDoubleComplex* dD_ks, cuDoubleComplex* vec_x, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N * 3)
    {
        vec_y[idx].x = dD_ks[idx % N].x * vec_x[idx].x - dD_ks[idx % N].y * vec_x[idx].y;
        vec_y[idx].y = dD_ks[idx % N].x * vec_x[idx].y + dD_ks[idx % N].y * vec_x[idx].x; 
    }
}

// matrix-vector multiplication A*x (a_H.*b)
static __global__ void DanMulti_conj(cuDoubleComplex* out, cuDoubleComplex *d_Dan, cuDoubleComplex *p, const int n){

    __shared__ cuDoubleComplex tmp1[BLOCK_SIZE], tmp2[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx<n){
        tmp1[threadIdx.x] = d_Dan[idx];
        tmp2[threadIdx.x] = p[idx];
        __syncthreads();
        out[idx].x = tmp1[threadIdx.x].x * tmp2[threadIdx.x].x + tmp1[threadIdx.x].y * tmp2[threadIdx.x].y;
        out[idx].y = tmp1[threadIdx.x].x * tmp2[threadIdx.x].y - tmp1[threadIdx.x].y * tmp2[threadIdx.x].x;
    }
}

// (x,y,z) -> (y,z,x)
static __global__ void Dan_Multi_trans(cuDoubleComplex* out, cuDoubleComplex* in, cuDoubleComplex* mtx, int n, int n1, int n2, int n3, int Gx, int Gz, double one_over_Gx, double one_over_Gz, int k2)
{   __shared__ cuDoubleComplex block[BLOCK_DIM_TR][BLOCK_DIM_TR+3];
    double tmp1;
    int s1, s2, t1, t2;
    int xIndex, yIndex, zIndex;
    int index_in, index_out;


    tmp1 = __uint2float_rz(blockIdx.x);
    tmp1 = floorf(tmp1 * one_over_Gz);
    s1 = __float2uint_rz(tmp1);
    t1 = blockIdx.x - Gz * s1;
    tmp1 = __uint2float_rz(blockIdx.y);
    tmp1 = floorf(tmp1 * one_over_Gx);
    s2 = __float2uint_rz(tmp1);
    t2 = blockIdx.y - Gx * s2;

    yIndex = s2*k2 + s1;

    zIndex   = t1 * BLOCK_DIM_TR + threadIdx.x;
    xIndex   = t2 * BLOCK_DIM_TR + threadIdx.y;
    if ((yIndex < n2) && (xIndex < n1) && (zIndex < n3)){
        index_in = (xIndex * n2 + yIndex) * n3 + zIndex;
        cuDoubleComplex input = in[index_in];
        cuDoubleComplex dan = mtx[index_in];
        block[threadIdx.y][threadIdx.x].x = input.x * dan.x - input.y * dan.y;
        block[threadIdx.y][threadIdx.x].y = input.y * dan.x + input.x * dan.y;
        //      block[threadIdx.y][threadIdx.x] = in[index_in];
    }
    __syncthreads();
    xIndex = t2 * BLOCK_DIM_TR + threadIdx.x;
    zIndex = t1 * BLOCK_DIM_TR + threadIdx.y;
    if ((yIndex < n2) && (xIndex < n1) && (zIndex < n3)){
        index_out = (yIndex * n3 + zIndex) * n1 + xIndex;
        out[index_out] = block[threadIdx.x][threadIdx.y];
    }

}
static __global__ void Dan_Multi_Complex_scale(cuDoubleComplex* out, cuDoubleComplex *d_Dan, cuDoubleComplex *p, int n){
    __shared__ cuDoubleComplex tmp1[BLOCK_SIZE], tmp2[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx<n){
        tmp1[threadIdx.x] = d_Dan[idx];
        tmp2[threadIdx.x] = p[idx];
        out[idx].x = (tmp1[threadIdx.x].x * tmp2[threadIdx.x].x - tmp1[threadIdx.x].y * tmp2[threadIdx.x].y)/n;
        out[idx].y = (tmp1[threadIdx.x].x * tmp2[threadIdx.x].y + tmp1[threadIdx.x].y * tmp2[threadIdx.x].x)/n;
    }

}

static __global__ void Conj_Dan_trans(cuDoubleComplex* out, cuDoubleComplex* in, cuDoubleComplex* mtx, int n, int n1, int n2, int n3, int Gx, int Gz, double one_over_Gx, double one_over_Gz, int k2)
{   __shared__ cuDoubleComplex block[BLOCK_DIM_TR][BLOCK_DIM_TR+2];
    float tmp1;
    int s1, s2, t1, t2;
    int xIndex, yIndex, zIndex;
    int index_in, index_out;


    tmp1 = __uint2float_rz(blockIdx.x);
    tmp1 = floorf(tmp1 * one_over_Gz);
    s1 = __float2uint_rz(tmp1);
    t1 = blockIdx.x - Gz * s1;
    tmp1 = __uint2float_rz(blockIdx.y);
    tmp1 = floorf(tmp1 * one_over_Gx);
    s2 = __float2uint_rz(tmp1);
    t2 = blockIdx.y - Gx * s2;
    yIndex = s2*k2 + s1;

    xIndex = t2 * BLOCK_DIM_TR + threadIdx.x;
    zIndex = t1 * BLOCK_DIM_TR + threadIdx.y;
    if ((yIndex < n2) && (xIndex < n1) && (zIndex < n3)){
        index_in = (yIndex * n3 + zIndex) *n1 + xIndex;
        block[threadIdx.y][threadIdx.x] = in[index_in];
    }
    __syncthreads();
    zIndex = t1 * BLOCK_DIM_TR + threadIdx.x;
    xIndex = t2 * BLOCK_DIM_TR + threadIdx.y;
    if ((yIndex < n2) && (xIndex < n1) && (zIndex < n3)){
        index_out = (xIndex*n2 + yIndex) * n3 + zIndex;
        out[index_out].x = mtx[index_out].x * block[threadIdx.x][threadIdx.y].x + mtx[index_out].y * block[threadIdx.x][threadIdx.y].y;
        out[index_out].y = mtx[index_out].x * block[threadIdx.x][threadIdx.y].y - mtx[index_out].y * block[threadIdx.x][threadIdx.y].x;
    }

}
static __global__ void Dan_Multi_transpose_1_1616(cuDoubleComplex *odata, cuDoubleComplex *idata, int n1, int n2, int n3, cuDoubleComplex* mtx)
{
    __shared__ cuDoubleComplex block[NN][NN+1];
    // read the matrix tile into shared memory
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int zIndex = blockIdx.z * blockDim.z + threadIdx.z;
//  if( xIndex < n1 && yIndex < n2 && zIndex < n3 )
    if( xIndex < n3 && yIndex < n2 && zIndex < n1 )
    {
//      uint index_in = zIndex * n1 * n2 + yIndex * n2 + xIndex;
//      uint index_in = zIndex * n1 * n2 + yIndex * n1 + xIndex;
        int index_in = zIndex * n3 * n2 + yIndex * n3 + xIndex;
        block[threadIdx.y][threadIdx.x].x = mtx[index_in].x * idata[index_in].x - mtx[index_in].y * idata[index_in].y;
        block[threadIdx.y][threadIdx.x].y = mtx[index_in].x * idata[index_in].y + mtx[index_in].y * idata[index_in].x;
        //      block[threadIdx.z][threadIdx.y][threadIdx.x] = idata[index_in];

    }
    __syncthreads();


    // write the transposed matrix tile to global memory
    xIndex = blockIdx.y * blockDim.y + threadIdx.x;
    yIndex = blockIdx.x * blockDim.x + threadIdx.y;
//  xIndex = blockIdx.x * blockDim.x + threadIdx.x;
//  yIndex = blockIdx.y * blockDim.y + threadIdx.y;
//  if( xIndex < n1 && yIndex <  n2 && zIndex < n3)
//  if( xIndex < n3 && yIndex <  n2 && zIndex < n1)
    if( xIndex < n2 && yIndex <  n3 && zIndex < n1)
    {
//      uint index_out = zIndex * n1 * n2 + yIndex * n1 + xIndex;
//      uint index_out = zIndex * n3 * n2 + yIndex * n3 + xIndex;
        int index_out = zIndex * n3 * n2 + yIndex * n2 + xIndex;
        odata[index_out] = block[threadIdx.x][threadIdx.y];
    }
}
static __global__ void ConjMulti_Conj_transpose_1_1616(cuDoubleComplex *odata, cuDoubleComplex *idata, int n1, int n2, int n3, cuDoubleComplex* mtx)
{
    __shared__ cuDoubleComplex block[NN][NN+1];
    // read the matrix tile into shared memory
//  uint xIndex = blockIdx.x * blockDim.x + threadIdx.x;
//  uint yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int xIndex = blockIdx.y * blockDim.y + threadIdx.x;
    int yIndex = blockIdx.x * blockDim.x + threadIdx.y;
    int zIndex = blockIdx.z * blockDim.z + threadIdx.z;
//  if( xIndex < n1 && yIndex < n2 && zIndex < n3 )
    if( xIndex < n2 && yIndex < n3 && zIndex < n1 )
    {
//      uint index_in = zIndex * n1 * n2 + yIndex * n2 + xIndex;
        int index_in = zIndex * n2 * n3 + yIndex * n2 + xIndex;
        block[threadIdx.y][threadIdx.x] = idata[index_in];

    }
    __syncthreads();


    // write the transposed matrix tile to global memory
//  xIndex = blockIdx.y * blockDim.y + threadIdx.x;
//  yIndex = blockIdx.x * blockDim.x + threadIdx.y;
    xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    yIndex = blockIdx.y * blockDim.y + threadIdx.y;
//  if( xIndex < n1 && yIndex <  n2 && zIndex < n3)
    if( xIndex < n3 && yIndex <  n2 && zIndex < n1)
    {
//      uint index_out = zIndex * n1 * n2 + yIndex * n1 + xIndex;
        int index_out = zIndex * n2 * n3 + yIndex * n3 + xIndex;
        odata[index_out].x = mtx[index_out].x * block[threadIdx.x][threadIdx.y].x + mtx[index_out].y * block[threadIdx.x][threadIdx.y].y;
        odata[index_out].y = mtx[index_out].x * block[threadIdx.x][threadIdx.y].y - mtx[index_out].y * block[threadIdx.x][threadIdx.y].x;
    }
}

