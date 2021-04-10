
///////////////////////////////////////////////////////////////
/// This code computes the inverse B under biisotorpic material.
///////////////////////////////////////////////////////////////

#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include <complex.h>
#include "printDeviceArray.cuh"

/*
Element : n = n1*n2*n3
	  length of ele_permitt_in = 100;
*/

static __global__ void alpha_ones(int N, realCPU* vec, realCPU alpha, realCPU beta, int* par_idx);
static __global__ void xi(int N, cmpxGPU* vec, realCPU rep_in, realCPU rep_out, realCPU chi_in, realCPU chi_out, int* par_idx);
static __global__ void zeta(int N, cmpxGPU* vec, realCPU rep_in, realCPU rep_out, realCPU chi_in, realCPU chi_out, int* par_idx);
static __global__ void create_invPhi(int size, realCPU* B_eps, realCPU* B_mu, cmpxGPU* B_xi, cmpxGPU* B_zeta, realCPU* inv_Phi);
static __global__ void conjugate(int size, cmpxGPU* ivec, cmpxGPU* ovec);

int FAME_Matrix_B_Biisotropic(int n, MATERIAL material, realCPU* B_eps, realCPU* B_mu, cmpxGPU* B_xi, cmpxGPU* B_zeta, cmpxGPU* B_zeta_s, realCPU* inv_Phi)
{
    printf("in FAME_Matrix_B_Biisotropic\n");
	int N_material_handle = material.material_num;
	/*int N_permitt     = material.num_ele_permitt_in;
	int N_permeab     = material.num_mag_permeab_in;
	int N_reciprocity = material.num_reciprocity_in;
	int N_chirality   = material.num_chirality_in;    
	
	if((N_material_handle != N_permitt)||(N_material_handle != N_permeab) || (N_material_handle != N_reciprocity) || (N_material_handle!=N_chirality))
	{
        cout << "The input number of material handle, permittivity and permeability not equal! Please check these input data." << endl;
	}*/

	int size = n*N_material_handle;
	int tn	 =3*n;
	int N = 3*size;// 3*Nx*Ny*Nz*t
	
	dim3 DimBlock(BLOCK_SIZE,1,1);
    dim3 DimGrid((N-1)/BLOCK_SIZE +1,1,1);	

	int* dB_inout;
	cudaMalloc((void**) &dB_inout, 2*N*sizeof(int));

	realCPU* temp_1 = (realCPU*)calloc(tn, sizeof(realCPU));

	for(int ii=0; ii<tn; ii++)
		temp_1[ii] = material.ele_permitt_out;
	
	//printf("material.ele_permitt_in % d \n",material.ele_permitt_in[0]);

	cudaMemcpy(dB_inout, material.Binout, 2*N*sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(B_eps,  temp_1, tn*sizeof(realCPU), cudaMemcpyHostToDevice);
	
	cudaMemcpy(B_mu,   temp_1, tn*sizeof(realCPU), cudaMemcpyHostToDevice);

	cmpxCPU* temp_2 = (cmpxCPU*)calloc(tn, sizeof(cmpxCPU));
	for(int ii=0; ii<tn; ii++)
		temp_2[ii] = material.reciprocity_out;
	
	cudaMemcpy(B_xi,   temp_2, tn*sizeof(cmpxGPU), cudaMemcpyHostToDevice);
	cudaMemcpy(B_zeta, temp_2, tn*sizeof(cmpxGPU), cudaMemcpyHostToDevice);

	
	for(int i = 0; i<N_material_handle; i++)
	{
		for(int j = 0; j<3; j++)
		{
				
    		alpha_ones<<<DimGrid, DimBlock>>>(n, B_eps+j*n, 
											  material.ele_permitt_in[i], material.ele_permitt_out,
											  dB_inout+i*n+j*n);

			cudaDeviceSynchronize();

			alpha_ones<<<DimGrid, DimBlock>>>(n, B_mu+j*n,
                                      		  material.mag_permeab_in[i], material.mag_permeab_out,
                                      		  dB_inout + i*n +j*n);

			cudaDeviceSynchronize();
			xi<<<DimGrid, DimBlock>>>(n, B_xi+j*n,
                              		  material.reciprocity_in[i],   material.reciprocity_out,
                              		  material.chirality_in[i],     material.chirality_out,
                              		  dB_inout + i*n +j*n);

			cudaDeviceSynchronize();
			zeta<<<DimGrid, DimBlock>>>(n, B_zeta+j*n,
                                material.reciprocity_in[i],   material.reciprocity_out,
                                material.chirality_in[i],     material.chirality_out,
                                dB_inout +i*n+j*n );
			cudaDeviceSynchronize();
		}
	}

	conjugate<<<DimBlock, DimGrid>>>(n, B_zeta, B_zeta_s);

	create_invPhi<<<DimBlock, DimGrid>>>(n, B_eps, B_mu, B_xi, B_zeta, inv_Phi);


	free(temp_1);
	free(temp_2);
	cudaFree(dB_inout);
    return 0;
}

static __global__ void alpha_ones(int N, realCPU* vec, realCPU alpha, realCPU beta, int* par_idx)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < N)
	{
		if(par_idx[idx] == 1)
			vec[idx] = alpha;
	}
}

static __global__ void xi(int N, cmpxGPU* vec, realCPU rep_in, realCPU rep_out, realCPU chi_in, realCPU chi_out, int* par_idx)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < N)
    {
		vec[idx].x += par_idx[idx]*rep_in;
        vec[idx].y += (-1)*(par_idx[idx]*chi_in);

    }
}

static __global__ void zeta(int N, cmpxGPU* vec, realCPU rep_in, realCPU rep_out, realCPU chi_in, realCPU chi_out, int* par_idx)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < N)
    {
		vec[idx].x += par_idx[idx]*rep_in;
        vec[idx].y += par_idx[idx]*chi_in;

    }

}

static __global__ void conjugate(int size, cmpxGPU* ivec, cmpxGPU* ovec)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if( idx < size )
    {
        ovec[idx*3].x =  ivec[idx*3].x;
        ovec[idx*3].y = -ivec[idx*3].y;
		
		ovec[idx*3+1].x =  ivec[idx*3+1].x;
        ovec[idx*3+1].y = -ivec[idx*3+1].y;

		ovec[idx*3+2].x =  ivec[idx*3+2].x;
        ovec[idx*3+2].y = -ivec[idx*3+2].y;
    }
}

static __global__ void create_invPhi(int size, realCPU* B_eps, realCPU* B_mu, cmpxGPU* B_xi, cmpxGPU* B_zeta, realCPU* inv_Phi)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if( idx < size )
    {
        realCPU tmp;
		tmp = B_eps[idx*3] - (B_xi[idx*3].x*B_zeta[idx*3].x - B_xi[idx*3].y*B_zeta[idx*3].y)/B_mu[idx*3];
        inv_Phi[idx*3] = 1.0/tmp;

		tmp = B_eps[idx*3+1] - (B_xi[idx*3+1].x*B_zeta[idx*3+1].x - B_xi[idx*3+1].y*B_zeta[idx*3+1].y)/B_mu[idx*3+1];
        inv_Phi[idx*3+1] = 1.0/tmp;

		tmp = B_eps[idx*3+2] - (B_xi[idx*3+2].x*B_zeta[idx*3+2].x - B_xi[idx*3+2].y*B_zeta[idx*3+2].y)/B_mu[idx*3+2];
        inv_Phi[idx*3+2] = 1.0/tmp;		


    }
}
