#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <assert.h>
#include <math.h>
#include <complex.h>
#include "CG_F.cuh"
#include "CG_dualF.cuh"
#include "printDeviceArray.cuh"

using namespace std;


int CG_FdualF(cmpxGPU *db,
    cmpxGPU *x,
    CULIB_HANDLES cuHandles, 
    FFT_BUFFER    fft_buffer,
	MTX_B mtx_B,
    int Nx,
    int Ny,
    int Nz,
    int Nd,
    int max_iter,
    realGPU tol,
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
	cmpxGPU* D_ks)
{

    int n = Nd * 16;
	int iter_CG_F, iter_CG_dualF, iter_CG;

	iter_CG_F = CG_F(db, x, cuHandles, fft_buffer, mtx_B, Nx, Ny, Nz, Nd, max_iter, tol, 
		Pi_Qr, Pi_Qrs, Pi_Qr_110, Pi_Qrs_110, Pi_Qr_101, Pi_Qrs_101, Pi_Qr_011, Pi_Qrs_011, D_k, D_ks);

    iter_CG_dualF = CG_dualF(db+n, x+n, cuHandles, fft_buffer, mtx_B, Nx, Ny, Nz, Nd, max_iter, tol, 
		Pi_Pr, Pi_Prs, Pi_Pr_110, Pi_Prs_110, Pi_Pr_101, Pi_Prs_101, Pi_Pr_011, Pi_Prs_011, D_k, D_ks);

    iter_CG = iter_CG_F > iter_CG_dualF ? iter_CG_F : iter_CG_dualF;
    

	return iter_CG;
	
}




int CG_FdualF(cmpxGPU *db,
    cmpxGPU *x,
    CULIB_HANDLES cuHandles, 
    FFT_BUFFER    fft_buffer,
	MTX_B mtx_B,
    int Nx,
    int Ny,
    int Nz,
    int Nd,
    int max_iter,
    realGPU tol,
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
    cmpxGPU* D_kx,
    cmpxGPU* D_ky,
	cmpxGPU* D_kz)
{

    int n = Nd * 16;
	int iter_CG_F, iter_CG_dualF;

	iter_CG_F = CG_F(db, x, cuHandles, fft_buffer, mtx_B, Nx, Ny, Nz, Nd, max_iter, tol, 
		Pi_Qr, Pi_Qrs, Pi_Qr_110, Pi_Qrs_110, Pi_Qr_101, Pi_Qrs_101, Pi_Qr_011, Pi_Qrs_011, D_kx, D_ky, D_kz);

    iter_CG_dualF = CG_dualF(db+n, x+n, cuHandles, fft_buffer, mtx_B, Nx, Ny, Nz, Nd, max_iter, tol, 
		Pi_Pr, Pi_Prs, Pi_Pr_110, Pi_Prs_110, Pi_Pr_101, Pi_Prs_101, Pi_Pr_011, Pi_Prs_011, D_kx, D_ky, D_kz);

    int iter_CG = iter_CG_F > iter_CG_dualF ? iter_CG_F : iter_CG_dualF;

	return iter_CG;
	
}