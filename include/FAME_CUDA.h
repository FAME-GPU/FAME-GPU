#ifndef _FAME_CUDA_H_
#define _FAME_CUDA_H_

#include <cufft.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <FAME_Use_Single.h>

#define BLOCK_SIZE 1024
#define BLOCK_DIM_TR 16
#define BATCH 1
#define TILE_DIM 32
#define BLOCK_ROWS 8
#define NUM_REPS 100
typedef unsigned int uint;

#if defined(USE_SINGLE)

typedef cuFloatComplex cmpxGPU;
typedef cufftComplex cmpxfft;
typedef float realGPU;

#define PC_cublas_dot cublasCdotc_v2
#define PC_cublas_axpy cublasCaxpy_v2
#define PC_cublas_nrm2 cublasScnrm2_v2
#define PC_cublas_dscal cublasCsscal_v2
#define PC_cublas_scal cublasCscal_v2
#define PC_cublas_rot cublasCsrot_v2
#define PC_cufft_type CUFFT_C2C
#define PC_cufft_Exec cufftExecC2C
#define PC_cublas_copy cublasCcopy_v2
#define PC_cublas_gemm cublasCgemm
#define PC_cublas_swap cublasCswap_v2
#define make_cucmpx make_cuFloatComplex 

#else

typedef cuDoubleComplex cmpxGPU;
typedef cufftDoubleComplex cmpxfft;
typedef double realGPU;

#define PC_cublas_dot cublasZdotc_v2
#define PC_cublas_axpy cublasZaxpy_v2
#define PC_cublas_nrm2 cublasDznrm2_v2
#define PC_cublas_dscal cublasZdscal_v2
#define PC_cublas_scal cublasZscal_v2
#define PC_cublas_rot cublasZdrot_v2
#define PC_cufft_type CUFFT_Z2Z
#define PC_cufft_Exec cufftExecZ2Z
#define PC_cublas_copy cublasZcopy_v2
#define PC_cublas_gemm cublasZgemm
#define PC_cublas_swap cublasZswap_v2
#define make_cucmpx make_cuDoubleComplex 

#endif

typedef struct
{
    cufftHandle      cufft_plan_1d_x;
    cufftHandle      cufft_plan_1d_y;
    cufftHandle      cufft_plan_1d_z;
    cufftHandle      cufft_plan;
	cublasHandle_t   cublas_handle;
    cmpxGPU* Nd2_temp1; // invAr tmp
    cmpxGPU* Nd2_temp2; // CG r
    cmpxGPU* Nd2_temp3; // CG p
    cmpxGPU* Nd2_temp4; // CG Ap
    cmpxGPU* N3_temp1;  // QBQ vec1, Qrs tmp
    cmpxGPU* N3_temp2;  // QBQ vec2, Qr  tmp
} CULIB_HANDLES;

typedef struct
{
    realGPU* Lambda_q_sqrt;
    cmpxGPU* dD_kx;
    cmpxGPU* dD_ky;
    cmpxGPU* dD_kz;
    cmpxGPU* dD_k;
    cmpxGPU* dD_ks;
    cmpxGPU* dPi_Qr;
    cmpxGPU* dPi_Pr;
    cmpxGPU* dPi_Qrs;
    cmpxGPU* dPi_Prs;
} LAMBDAS_CUDA;

typedef struct
{
	realGPU* invB_eps;
	realGPU* B_eps;
    realGPU* invPhi;
    realGPU* B_mu;
    cmpxGPU* B_zeta;
    cmpxGPU* B_zeta_s;
    cmpxGPU* B_xi;
} MTX_B;

typedef struct
{
	cmpxGPU* d_A;
    cmpxfft* dvec_x;
	cmpxGPU* tmp;
} FFT_BUFFER;

typedef struct
{
	cmpxGPU* dU;
    cmpxGPU* dz;
    cmpxCPU*   z;  // ritz_vec
    realGPU* T0; // diag(T, 0)  Talpha
    realGPU* T1; // diag(T, 1)  Tbeta
    realGPU* T2; // diag(R, 2)  uu
    realGPU* T3; // temp T3    utemp
    realGPU* LT0; // Lapack T0  T_d
    realGPU* LT1; // Lapack T1 T_e
} LANCZOS_BUFFER;

typedef struct
{
	realGPU* time;
	int*	inner_iter; // for SIRA
	realGPU* outer_norm_r; // for SIRA
	realGPU  fft_time;
	realGPU  ifft_time;
} TIME_ELAPSED;

#endif
