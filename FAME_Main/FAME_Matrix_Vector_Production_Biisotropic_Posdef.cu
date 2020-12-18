#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include "printDeviceArray.cuh"


/*
size of vec_y = 4*Nd
size of vec_x = 4*Nd
Update  : yilin
Date    : 2018/11/21
*/


static __global__ void i_inv_sigma_r(int size, realGPU* L_q_s, cmpxGPU* ivec, cmpxGPU* ovec);

static __global__ void i_m_inv_sigma_r(int size, realGPU* L_q_s, cmpxGPU* ivec, cmpxGPU* ovec);

void FAME_Matrix_Vector_Production_Biisotropic_Posdef
	(	CULIB_HANDLES cuHandles,
	 	cmpxGPU* vec_x,
		MTX_B mtx_B,
		int Nx,
        int Ny,
        int Nz,
        int Nd,
		realGPU* Lambda_q_sqrt,
        cmpxGPU* Pi_Qr,
        cmpxGPU* Pi_Pr,
        cmpxGPU* Pi_Qrs,
        cmpxGPU* Pi_Prs,
		cmpxGPU* D_k,
		cmpxGPU* D_ks,
		cmpxGPU* vec_y)
{
	
	dim3 DimBlock(BLOCK_SIZE,1,1);
    dim3 DimGrid((Nd-1)/BLOCK_SIZE +1,1,1);

	i_inv_sigma_r<<<DimGrid, DimBlock>>>(Nd, Lambda_q_sqrt, vec_x+2*Nd, vec_y);
	i_m_inv_sigma_r<<<DimGrid, DimBlock>>>(Nd, Lambda_q_sqrt, vec_x, vec_y+2*Nd);

}

static __global__ void i_inv_sigma_r(int size, realGPU* L_q_s, cmpxGPU* ivec, cmpxGPU* ovec)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if( idx < size )
	{
		ovec[idx].y = ivec[idx].x/L_q_s[idx];
		ovec[idx].x = - ivec[idx].y/L_q_s[idx];
		ovec[idx+size].y = ivec[idx+size].x/L_q_s[idx];
        ovec[idx+size].x = - ivec[idx+size].y/L_q_s[idx];
	}
}

static __global__ void i_m_inv_sigma_r(int size, realGPU* L_q_s, cmpxGPU* ivec, cmpxGPU* ovec)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if( idx < size )
    {
		ovec[idx].y = -ivec[idx].x/L_q_s[idx];
        ovec[idx].x = ivec[idx].y/L_q_s[idx];
		ovec[idx+size].y = -ivec[idx+size].x/L_q_s[idx];
        ovec[idx+size].x = ivec[idx+size].y/L_q_s[idx];
	}

}
