
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

static __global__ void alpha_ones(int N, double* vec, double alpha, double beta, int* par_idx);
static __global__ void xi(int N, cuDoubleComplex* vec, double rep_in, double rep_out, double chi_in, double chi_out, int* par_idx);
static __global__ void zeta(int N, cuDoubleComplex* vec, double rep_in, double rep_out, double chi_in, double chi_out, int* par_idx);
static __global__ void create_invPhi(int size, double* B_eps, double* B_mu, cuDoubleComplex* B_xi, cuDoubleComplex* B_zeta, double* inv_Phi);
static __global__ void conjugate(int size, cuDoubleComplex* ivec, cuDoubleComplex* ovec);

int FAME_Matrix_B_Biisotropic(int n, MATERIAL material, double* B_eps, double* B_mu, cuDoubleComplex* B_xi, cuDoubleComplex* B_zeta, cuDoubleComplex* B_zeta_s, double* inv_Phi)
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
    dim3 DimGrid((n-1)/BLOCK_SIZE +1,1,1);	

	int* dB_inout;
	cudaMalloc((void**) &dB_inout, 2*N*sizeof(int));

	double* temp_1 = (double*)calloc(tn, sizeof(double));

	for(int ii=0; ii<tn; ii++)
		temp_1[ii] = material.ele_permitt_out;
	
	//printf("material.ele_permitt_in % d \n",material.ele_permitt_in[0]);

	cudaMemcpy(dB_inout, material.Binout, 2*N*sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(B_eps,  temp_1, tn*sizeof(double), cudaMemcpyHostToDevice);
	
	cudaMemcpy(B_mu,   temp_1, tn*sizeof(double), cudaMemcpyHostToDevice);

	cmpx* temp_2 = (cmpx*)calloc(tn, sizeof(cmpx));
	for(int ii=0; ii<tn; ii++)
		temp_2[ii] = material.reciprocity_out;
	
	cudaMemcpy(B_xi,   temp_2, tn*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(B_zeta, temp_2, tn*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

	
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

static __global__ void alpha_ones(int N, double* vec, double alpha, double beta, int* par_idx)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < N)
	{
		if(par_idx[idx] == 1)
			vec[idx] = alpha;
	}
}

static __global__ void xi(int N, cuDoubleComplex* vec, double rep_in, double rep_out, double chi_in, double chi_out, int* par_idx)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < N)
    {
		vec[idx].x += par_idx[idx]*rep_in;
        vec[idx].y += (-1)*(par_idx[idx]*chi_in);

    }
}

static __global__ void zeta(int N, cuDoubleComplex* vec, double rep_in, double rep_out, double chi_in, double chi_out, int* par_idx)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < N)
    {
		vec[idx].x += par_idx[idx]*rep_in;
        vec[idx].y += par_idx[idx]*chi_in;

    }

}

static __global__ void conjugate(int size, cuDoubleComplex* ivec, cuDoubleComplex* ovec)
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

static __global__ void create_invPhi(int size, double* B_eps, double* B_mu, cuDoubleComplex* B_xi, cuDoubleComplex* B_zeta, double* inv_Phi)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if( idx < size )
    {
        double tmp;
		tmp = B_eps[idx*3] - (B_xi[idx*3].x*B_zeta[idx*3].x - B_xi[idx*3].y*B_zeta[idx*3].y)/B_mu[idx*3];
        inv_Phi[idx*3] = 1.0/tmp;

		tmp = B_eps[idx*3+1] - (B_xi[idx*3+1].x*B_zeta[idx*3+1].x - B_xi[idx*3+1].y*B_zeta[idx*3+1].y)/B_mu[idx*3+1];
        inv_Phi[idx*3+1] = 1.0/tmp;

		tmp = B_eps[idx*3+2] - (B_xi[idx*3+2].x*B_zeta[idx*3+2].x - B_xi[idx*3+2].y*B_zeta[idx*3+2].y)/B_mu[idx*3+2];
        inv_Phi[idx*3+2] = 1.0/tmp;		


    }
}
