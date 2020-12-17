#ifndef _FAME_CUDA_H_
#define _FAME_CUDA_H_

#include <cufft.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define BLOCK_SIZE 1024
#define BLOCK_DIM_TR 16
#define BATCH 1
#define TILE_DIM 32
#define BLOCK_ROWS 8
#define NUM_REPS 100
typedef unsigned int uint;
typedef struct
{
    cufftHandle      cufft_plan_1d_x;
    cufftHandle      cufft_plan_1d_y;
    cufftHandle      cufft_plan_1d_z;
    cufftHandle      cufft_plan;
	cublasHandle_t   cublas_handle;
    cuDoubleComplex* Nd2_temp1; // invAr tmp
    cuDoubleComplex* Nd2_temp2; // CG r
    cuDoubleComplex* Nd2_temp3; // CG p
    cuDoubleComplex* Nd2_temp4; // CG Ap
    cuDoubleComplex* N3_temp1;  // QBQ vec1, Qrs tmp
    cuDoubleComplex* N3_temp2;  // QBQ vec2, Qr  tmp
} CULIB_HANDLES;

typedef struct
{
    double* Lambda_q_sqrt;
    cuDoubleComplex* dD_kx;
    cuDoubleComplex* dD_ky;
    cuDoubleComplex* dD_kz;
    cuDoubleComplex* dD_k;
    cuDoubleComplex* dD_ks;
    cuDoubleComplex* dPi_Qr;
    cuDoubleComplex* dPi_Pr;
    cuDoubleComplex* dPi_Qrs;
    cuDoubleComplex* dPi_Prs;
} LAMBDAS_CUDA;

typedef struct
{
	double* invB_eps;
	double* B_eps;
    double* invPhi;
    double* B_mu;
    cuDoubleComplex* B_zeta;
    cuDoubleComplex* B_zeta_s;
    cuDoubleComplex* B_xi;
} MTX_B;

typedef struct
{
	cuDoubleComplex* d_A;
    cufftDoubleComplex* dvec_x;
	cuDoubleComplex* tmp;
} FFT_BUFFER;

typedef struct
{
	cuDoubleComplex* dU;
    cuDoubleComplex* dz;
    cmpx*   z;  // ritz_vec
    double* T0; // diag(T, 0)  Talpha
    double* T1; // diag(T, 1)  Tbeta
    double* T2; // diag(R, 2)  uu
    double* T3; // temp T3    utemp
    double* LT0; // Lapack T0  T_d
    double* LT1; // Lapack T1 T_e
    double* c;
    double* s;
} LANCZOS_BUFFER;

typedef struct
{
	double* time;
	int*	inner_iter; // for SIRA
	double* outer_norm_r; // for SIRA
	double  fft_time;
	double  ifft_time;
} TIME_ELAPSED;

#endif
