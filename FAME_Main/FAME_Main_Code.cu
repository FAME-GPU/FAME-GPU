#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include "FAME_FFT_CUDA.cuh"
#include "FAME_Create_cublas.cuh"
#include "FAME_Create_Buffer_Isotropic.cuh"
#include "FAME_Create_Buffer_Anisotropic.cuh"
#include "FAME_Create_Buffer_Biisotropic.cuh"
#include "FAME_Create_Buffer_Bianisotropic.cuh"
#include "FAME_Matrix_B_Isotropic.cuh"
#include "FAME_Matrix_B_Anisotropic.cuh"
#include "FAME_Matrix_B_Biisotropic.cuh"
#include "FAME_Matrix_B_Bianisotropic.cuh"
#include "FAME_Malloc_mtx_C.h"
#include "FAME_Matrix_Lambdas.cuh"
#include "FAME_Matrix_Curl.h"
#include "FAME_Create_Frequency_txt.h"
#include "FAME_Save_Eigenvector.h"
#include "FAME_Profile.h"
#include "FAME_Destroy_Main.cuh"
#include "FAME_Fast_Algorithms_Isotropic.cuh"
#include "FAME_Fast_Algorithms_Anisotropic.cuh"
#include "FAME_Fast_Algorithms_Biisotropic.cuh"
#include "FAME_Fast_Algorithms_Bianisotropic.cuh"
#include "FAME_Matrix_Vector_Production_Qrs.cuh"
#include "FAME_Matrix_Vector_Production_Pr.cuh"
#include "FAME_Print_Parameter.h"
#include "FAME_Create_Lambdas_txt.h"
#include "vec_plus.h"
#include "vec_norm.h"
#include "vec_inner_prod.h"
#include "mtx_print.h"
#include "mtx_prod.h"
#include "mtx_trans.h"
#include "mtx_trans_conj.h"
#include "mtx_cat.h"
#include "mtx_dot_prod.h"
#include "kron_vec.h"
#include "inv3.h"
#include "printDeviceArray.cuh"

void FAME_Fast_Algorithms_Driver(
	realCPU*        Freq_array,
	cmpxCPU*        Ele_field_mtx,
	cmpxCPU*        Dis_field_mtx,
	CULIB_HANDLES   cuHandles,
	LANCZOS_BUFFER  lBuffer,
	FFT_BUFFER      fft_buffer,
	LAMBDAS_CUDA    Lambdas_cuda,
	MTX_B           mtx_B,
	MATERIAL 	    material,
	ES              es,
	LS              ls,
	int Nx, int Ny, int Nz, int Nd, int N, 
	char* lattice_type, PROFILE* Profile);

void Destroy_Lambdas(LAMBDAS Lambdas, LAMBDAS_CUDA Lambdas_cuda, char* lattice_type);
void Check_Eigendecomp(MTX_C mtx_C, LAMBDAS Lambdas, LAMBDAS_CUDA Lambdas_cuda, FFT_BUFFER fft_buffer, CULIB_HANDLES cuHandles,
	int Nx, int Ny, int Nz, int Nd, int N, char* lattice_type, realCPU error, PROFILE* Profile);
void Check_Residual_Iso(realCPU* Freq_array, cmpxCPU* Ele_field_mtx, MTX_B mtx_B, MTX_C mtx_C, int N, int Nwant, realCPU error);
void Check_Residual_Aniso(realCPU* Freq_array, cmpxCPU* Ele_field_mtx, cmpxCPU* Dis_field_mtx, MTX_B mtx_B, MTX_C mtx_C, int N, int Nwant, realCPU error);
void Check_Residual_Biiso(realCPU* Freq_array, cmpxCPU* Ele_field_mtx, MTX_B mtx_B, MTX_C mtx_C, int N, int Nwant, realCPU error);
void Check_Residual_Bianiso(realCPU* Freq_array, cmpxCPU* Ele_field_mtx, cmpxCPU* Dis_field_mtx, MTX_B mtx_B, MTX_C mtx_C, int N, int Nwant, realCPU error);

int FAME_Main_Code(PAR Par, PROFILE* Profile)
{

	int Nx = Par.mesh.grid_nums[0];
    int Ny = Par.mesh.grid_nums[1];
	int Nz = Par.mesh.grid_nums[2];
	int Nd;
	int N  = Nx * Ny * Nz;
	int Ele_field_mtx_N = 0;
	int Nwant = Par.es.nwant;
	int Nstep = Par.es.nstep;
	int N_wave_vec = Par.recip_lattice.Wave_vec_num;
	realCPU wave_vec_array[3];
	
	#if defined(USE_SINGLE)
		Par.ce_error = 1e-1;
		Par.es.tol = 1e-6;
		Par.ls.tol = 1e-6;
		realCPU cr_error = 1e-3;
	#else
		Par.ce_error = 1e-7;
		Par.es.tol = 1e-12;
		Par.ls.tol = 1e-12;
		realCPU cr_error = 1e-8;
	#endif 

	struct timespec start, end;
	realCPU accum;

	cudaSetDevice(Par.flag.device);
	
    CULIB_HANDLES  cuHandles;
	FFT_BUFFER     fft_buffer;
	LANCZOS_BUFFER lBuffer;
	MTX_B          mtx_B;
	MTX_C          mtx_C;
	LAMBDAS        Lambdas;
    LAMBDAS_CUDA   Lambdas_cuda;

	FAME_Create_cublas(&cuHandles, Nx, Ny, Nz);

	realCPU *Freq_array;
	cmpxCPU *Ele_field_mtx, *Dis_field_mtx;
	if(strcmp(Par.material.material_type, "isotropic") == 0)
	{
		Ele_field_mtx_N = N * 3;
		Freq_array = (realCPU*) calloc(N_wave_vec * Nwant, sizeof(realCPU));
		Ele_field_mtx = (cmpxCPU*) calloc(Ele_field_mtx_N * Nwant, sizeof(cmpxCPU));
	}
	else if(strcmp(Par.material.material_type, "anisotropic") == 0)
	{
		Ele_field_mtx_N = N * 12;
		Freq_array = (realCPU*) calloc(N_wave_vec * Nwant, sizeof(realCPU));
		Ele_field_mtx = (cmpxCPU*) calloc(Ele_field_mtx_N * Nwant, sizeof(cmpxCPU));
		Dis_field_mtx = (cmpxCPU*) calloc(Ele_field_mtx_N * Nwant, sizeof(cmpxCPU));
	}
	else if(strcmp(Par.material.material_type, "biisotropic") == 0)
	{
		Ele_field_mtx_N = N * 6;
		Freq_array = (realCPU*) calloc(N_wave_vec * Nwant, sizeof(realCPU));
		Ele_field_mtx = (cmpxCPU*) calloc(Ele_field_mtx_N * Nwant, sizeof(cmpxCPU));
	}
	else if(strcmp(Par.material.material_type, "bianisotropic") == 0)
	{
		Ele_field_mtx_N = N * 48;
		Freq_array = (realCPU*) calloc(N_wave_vec * Nwant, sizeof(realCPU));
		Ele_field_mtx = (cmpxCPU*) calloc(Ele_field_mtx_N * Nwant, sizeof(cmpxCPU));
		Dis_field_mtx = (cmpxCPU*) calloc(Ele_field_mtx_N * Nwant, sizeof(cmpxCPU));
	}
	
	if(strcmp(Par.material.material_type, "isotropic") == 0)
	{
		FAME_Create_Buffer_Isotropic(&cuHandles, &fft_buffer, &lBuffer, N, Nstep);
	}
	else if(strcmp(Par.material.material_type, "anisotropic") == 0)
	{
		FAME_Create_Buffer_Anisotropic(&cuHandles, &fft_buffer, &lBuffer, N, Nstep);
	}
	else if(strcmp(Par.material.material_type, "biisotropic") == 0)
	{
		FAME_Create_Buffer_Biisotropic(&cuHandles, &fft_buffer, &lBuffer, N, Nstep);
	}
	else if(strcmp(Par.material.material_type, "bianisotropic") == 0)
	{
		FAME_Create_Buffer_Bianisotropic(&cuHandles, &fft_buffer, &lBuffer, N, Nstep);
	}

	
	if(strcmp(Par.material.material_type, "isotropic") == 0)
	{
		printf("= = = = FAME_Matrix_B_Isotropic = = = = = = = = = = = = = = = = = = = = = = = = =\n");
		size_t size = 3*N*sizeof(realCPU);
		checkCudaErrors(cudaMalloc((void**) &mtx_B.B_eps,    size));
		checkCudaErrors(cudaMalloc((void**) &mtx_B.invB_eps, size));
		FAME_Matrix_B_Isotropic(mtx_B.B_eps, mtx_B.invB_eps, Par.material, N);
	}

	else if(strcmp(Par.material.material_type, "anisotropic") == 0)
    {
		printf("= = = = FAME_Matrix_B_Anisotropic = = = = = = = = = = = = = = = = = = = = = = = = =\n");
		cudaMalloc((void**) &mtx_B.N, 9 * sizeof(cmpxGPU));
		cudaMalloc((void**) &mtx_B.GInOut_index,  Par.material.BInOut_index_length[7] * sizeof(int));
		mtx_B.GInOut_index_length = (int *) calloc(8, sizeof(int));
    	cudaMemcpy(mtx_B.GInOut_index, Par.material.BInOut_index, Par.material.BInOut_index_length[7] * sizeof(int), cudaMemcpyHostToDevice);
    	memcpy(mtx_B.GInOut_index_length, Par.material.BInOut_index_length, 8 * sizeof(int));

		FAME_Matrix_B_Anisotropic(N, Par.material, mtx_B.N );
	}

	else if(strcmp(Par.material.material_type, "biisotropic") == 0)
    {
		printf("= = = = FAME_Matrix_B_Biisotropic = = = = = = = = = = = = = = = = = = = = = = = = =\n");
		size_t size = 3 * N * sizeof(realGPU);
		cudaMalloc((void**) &mtx_B.B_eps, size);
		cudaMalloc((void**) &mtx_B.B_mu, size);
		cudaMalloc((void**) &mtx_B.invPhi, size);

		size = 3 * N * sizeof(cmpxGPU);
		cudaMalloc((void**) &mtx_B.B_zeta, size);
		cudaMalloc((void**) &mtx_B.B_zeta_s, size);
		cudaMalloc((void**) &mtx_B.B_xi, size);

		FAME_Matrix_B_Biisotropic(N, Par.material, mtx_B.B_eps, mtx_B.B_mu, mtx_B.B_xi, mtx_B.B_zeta, mtx_B.B_zeta_s, mtx_B.invPhi);
	}

	else if(strcmp(Par.material.material_type, "bianisotropic") == 0)
    {
		printf("= = = = FAME_Matrix_B_Bianisotropic = = = = = = = = = = = = = = = = = = = = = = = = =\n");
		cudaMalloc((void**) &mtx_B.G, 36 * sizeof(cmpxGPU));
		cudaMalloc((void**) &mtx_B.GInOut_index,  Par.material.BInOut_index_length[7] * sizeof(int));
		mtx_B.GInOut_index_length = (int *) calloc(8, sizeof(int));
    	cudaMemcpy(mtx_B.GInOut_index, Par.material.BInOut_index, Par.material.BInOut_index_length[7] * sizeof(int), cudaMemcpyHostToDevice);
    	memcpy(mtx_B.GInOut_index_length, Par.material.BInOut_index_length, 8 * sizeof(int));

		FAME_Matrix_B_Bianisotropic(N, Par.material, mtx_B.G);
	}


    FAME_Malloc_mtx_C(&mtx_C, N);
	FAME_Print_Parameter(Par);
	for(int i = 0; i < N_wave_vec; i++)
	{
		Profile->idx = i;

	    wave_vec_array[0] = Par.recip_lattice.WaveVector[3 * i];
    	wave_vec_array[1] = Par.recip_lattice.WaveVector[3 * i + 1];
    	wave_vec_array[2] = Par.recip_lattice.WaveVector[3 * i + 2];

    	printf("\033[40;33m= = Start to compute (%3d/%3d) WaveVector = [ % .6f % .6f % .6f ] = =\033[0m\n", i + 1, Par.recip_lattice.Wave_vec_num, wave_vec_array[0], wave_vec_array[1], wave_vec_array[2]);

		printf("= = = = FAME_Matrix_Curl  = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n");
		FAME_Matrix_Curl(&mtx_C, wave_vec_array, Par.mesh.grid_nums, Par.mesh.edge_len, Par.mesh.mesh_len, Par.lattice);

		printf("= = = = FAME_Matrix_Lambdas = = = = = = = = = = = = = = = = = = = = = = = = = = =\n");
		Nd = FAME_Matrix_Lambdas(&Lambdas_cuda, wave_vec_array, Par.mesh.grid_nums, Par.mesh.mesh_len, Par.lattice.lattice_vec_a, &Par, &Lambdas);
		
		printf("= = = = Check_Eigendecomp = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n");
		clock_gettime(CLOCK_REALTIME, &start);
		Check_Eigendecomp(mtx_C, Lambdas, Lambdas_cuda, fft_buffer, cuHandles, Nx, Ny, Nz, Nd, N, Par.lattice.lattice_type, Par.ce_error, Profile);
		clock_gettime(CLOCK_REALTIME, &end);
		accum = ( end.tv_sec - start.tv_sec ) + ( end.tv_nsec - start.tv_nsec ) / BILLION;
		printf("%*s%8.2f sec.\n", 68, "", accum);

		printf("= = = = FAME_Fast_Algorithms = = = = = = = = = = = = = = = = = = = = = = = = = = =\n");
		clock_gettime (CLOCK_REALTIME, &start);
		FAME_Fast_Algorithms_Driver(Freq_array + i * Nwant, Ele_field_mtx, Dis_field_mtx, 
			cuHandles, lBuffer, fft_buffer, Lambdas_cuda, mtx_B, Par.material, Par.es, Par.ls,
			Nx, Ny, Nz, Nd, N, Par.lattice.lattice_type, Profile);
		clock_gettime (CLOCK_REALTIME, &end);
		Profile->es_time[Profile->idx] = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / BILLION;
		
		printf("= = = = Check_Residual  = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n");
		clock_gettime (CLOCK_REALTIME, &start);
		if(strcmp(Par.material.material_type, "isotropic") == 0)
		{
			Check_Residual_Iso(Freq_array+i*Nwant, Ele_field_mtx, mtx_B, mtx_C, N, Nwant, cr_error);
		}
		else if(strcmp(Par.material.material_type, "anisotropic") == 0)
		{
			Check_Residual_Aniso(Freq_array+i*Nwant, Ele_field_mtx, Dis_field_mtx, mtx_B, mtx_C, N, Nwant, cr_error);
		}
		else if(strcmp(Par.material.material_type, "biisotropic") == 0)
		{
			Check_Residual_Biiso(Freq_array+i*Nwant, Ele_field_mtx, mtx_B, mtx_C, N, Nwant, cr_error);
		}
		else if(strcmp(Par.material.material_type, "bianisotropic") == 0)
		{
			Check_Residual_Bianiso(Freq_array+i*Nwant, Ele_field_mtx, Dis_field_mtx, mtx_B, mtx_C, N, Nwant, cr_error);
		}
		clock_gettime (CLOCK_REALTIME, &end);
		accum = ( end.tv_sec - start.tv_sec ) + ( end.tv_nsec - start.tv_nsec ) / BILLION;
		printf("%*s%8.2f sec.\n", 68, "", accum);
		

		if(Par.flag.save_eigen_vector)
		{
			printf("= = = = Save Eigen Vector = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n");
			FAME_Save_Eigenvector(Ele_field_mtx, Nwant, Ele_field_mtx_N, i);
		}

		Destroy_Lambdas(Lambdas, Lambdas_cuda, Par.lattice.lattice_type);

		FAME_Print_Profile(*Profile);

	}

	FAME_Create_Frequency_txt(Freq_array, Nwant, Profile->idx);
	
	FAME_Destroy_Main(cuHandles, fft_buffer, lBuffer, mtx_B, mtx_C, Freq_array, Ele_field_mtx);
	
	return 0;
}

void FAME_Fast_Algorithms_Driver(
	realCPU*        Freq_array,
	cmpxCPU*        Ele_field_mtx,
	cmpxCPU*        Dis_field_mtx, 
	CULIB_HANDLES  cuHandles,
	LANCZOS_BUFFER lBuffer,
	FFT_BUFFER     fft_buffer,
	LAMBDAS_CUDA   Lambdas_cuda,
	MTX_B          mtx_B,
	MATERIAL 	   material,
	ES             es,
	LS             ls,
	int Nx, int Ny, int Nz, int Nd, int N, 
	char* lattice_type, PROFILE* Profile)
{

	if(strcmp(material.material_type, "isotropic") == 0)
	{
		if((strcmp(lattice_type, "simple_cubic"          ) == 0) || \
		   (strcmp(lattice_type, "primitive_orthorhombic") == 0) || \
		   (strcmp(lattice_type, "primitive_tetragonal"  ) == 0))
		{

			FAME_Fast_Algorithms_Isotropic(Freq_array, Ele_field_mtx, cuHandles, lBuffer, fft_buffer,
								    Lambdas_cuda, mtx_B, es, ls, Nx, Ny, Nz, Nd, N, "Simple", Profile);
		}
		else
		{
			FAME_Fast_Algorithms_Isotropic(Freq_array, Ele_field_mtx, cuHandles, lBuffer, fft_buffer,
								 	Lambdas_cuda, mtx_B, es, ls, Nx, Ny, Nz, Nd, N, "General", Profile);
		}
	}
	else if (strcmp(material.material_type, "anisotropic") == 0)
	{
		if( (strcmp(lattice_type, "simple_cubic"          ) == 0) || \
			(strcmp(lattice_type, "primitive_orthorhombic") == 0) || \
			(strcmp(lattice_type, "primitive_tetragonal"  ) == 0) )
		{		
			FAME_Fast_Algorithms_Anisotropic(Freq_array, Ele_field_mtx, Dis_field_mtx, cuHandles, lBuffer, fft_buffer,
									Lambdas_cuda, mtx_B, es, ls, Nx, Ny, Nz, Nd, N, "Simple", Profile);		
		}
		else
		{
			FAME_Fast_Algorithms_Anisotropic(Freq_array, Ele_field_mtx, Dis_field_mtx, cuHandles, lBuffer, fft_buffer,
									Lambdas_cuda, mtx_B, es, ls, Nx, Ny, Nz, Nd, N, "General", Profile);			
		}
	}
	else if (strcmp(material.material_type, "biisotropic") == 0)
	{
		if( (strcmp(lattice_type, "simple_cubic"          ) == 0) || \
			(strcmp(lattice_type, "primitive_orthorhombic") == 0) || \
			(strcmp(lattice_type, "primitive_tetragonal"  ) == 0) )
		{		
			FAME_Fast_Algorithms_Biisotropic(Freq_array, Ele_field_mtx, cuHandles, Lambdas_cuda, lBuffer, fft_buffer, 
									mtx_B, material, Nx, Ny, Nz, Nd, es, ls, "Simple", Profile);		
		}
		else
		{
			FAME_Fast_Algorithms_Biisotropic(Freq_array, Ele_field_mtx, cuHandles,Lambdas_cuda, lBuffer, fft_buffer, 
									mtx_B, material, Nx, Ny, Nz, Nd, es, ls, "General", Profile);			
		}
	}
	else if (strcmp(material.material_type, "bianisotropic") == 0)
	{
		if( (strcmp(lattice_type, "simple_cubic"          ) == 0) || \
			(strcmp(lattice_type, "primitive_orthorhombic") == 0) || \
			(strcmp(lattice_type, "primitive_tetragonal"  ) == 0) )
		{		
			FAME_Fast_Algorithms_Bianisotropic(Freq_array, Ele_field_mtx, Dis_field_mtx, cuHandles, Lambdas_cuda, lBuffer, fft_buffer, 
									mtx_B, material ,Nx, Ny, Nz, Nd, es, ls, "Simple", Profile);		
		}
		else
		{
			FAME_Fast_Algorithms_Bianisotropic(Freq_array, Ele_field_mtx, Dis_field_mtx, cuHandles, Lambdas_cuda, lBuffer, fft_buffer, 
									mtx_B, material ,Nx, Ny, Nz, Nd, es, ls , "General", Profile);			
		}
	}
}

void Destroy_Lambdas(LAMBDAS Lambdas, LAMBDAS_CUDA Lambdas_cuda, char* lattice_type)
{
	if((strcmp(lattice_type, "simple_cubic"          ) == 0) || \
	   (strcmp(lattice_type, "primitive_orthorhombic") == 0) || \
	   (strcmp(lattice_type, "primitive_tetragonal"  ) == 0))
	{
        free(Lambdas.D_k);
        free(Lambdas.D_ks);
        free(Lambdas.Lambda_x);
        free(Lambdas.Lambda_y);
        free(Lambdas.Lambda_z);

		cudaFree(Lambdas_cuda.dD_k);
		cudaFree(Lambdas_cuda.dD_ks);
	}

	else
	{
        free(Lambdas.D_kx);
        free(Lambdas.D_ky);
        free(Lambdas.D_kz);
        free(Lambdas.Lambda_x);
        free(Lambdas.Lambda_y);
        free(Lambdas.Lambda_z);

		cudaFree(Lambdas_cuda.dD_kx);
    	cudaFree(Lambdas_cuda.dD_ky);
    	cudaFree(Lambdas_cuda.dD_kz);
	}

    free(Lambdas.Lambda_q_sqrt);
    free(Lambdas.Pi_Qr);
    free(Lambdas.Pi_Pr);
    free(Lambdas.Pi_Qrs);
    free(Lambdas.Pi_Prs);
	free(Lambdas.Pi_Qr_110);
    free(Lambdas.Pi_Pr_110);
    free(Lambdas.Pi_Qrs_110);
    free(Lambdas.Pi_Prs_110);
	free(Lambdas.Pi_Qr_101);
    free(Lambdas.Pi_Pr_101);
    free(Lambdas.Pi_Qrs_101);
    free(Lambdas.Pi_Prs_101);
	free(Lambdas.Pi_Qr_011);
    free(Lambdas.Pi_Pr_011);
    free(Lambdas.Pi_Qrs_011);
    free(Lambdas.Pi_Prs_011);

    cudaFree(Lambdas_cuda.Lambda_q_sqrt);
	cudaFree(Lambdas_cuda.dPi_Qr);
	cudaFree(Lambdas_cuda.dPi_Pr);
	cudaFree(Lambdas_cuda.dPi_Qrs);
	cudaFree(Lambdas_cuda.dPi_Prs);
	cudaFree(Lambdas_cuda.dPi_Qr_110);
	cudaFree(Lambdas_cuda.dPi_Pr_110);
	cudaFree(Lambdas_cuda.dPi_Qrs_110);
	cudaFree(Lambdas_cuda.dPi_Prs_110);
	cudaFree(Lambdas_cuda.dPi_Qr_101);
	cudaFree(Lambdas_cuda.dPi_Pr_101);
	cudaFree(Lambdas_cuda.dPi_Qrs_101);
	cudaFree(Lambdas_cuda.dPi_Prs_101);
	cudaFree(Lambdas_cuda.dPi_Qr_011);
	cudaFree(Lambdas_cuda.dPi_Pr_011);
	cudaFree(Lambdas_cuda.dPi_Qrs_011);
	cudaFree(Lambdas_cuda.dPi_Prs_011);
}

void Check_Eigendecomp(MTX_C mtx_C, LAMBDAS Lambdas, LAMBDAS_CUDA Lambdas_cuda, FFT_BUFFER fft_buffer, CULIB_HANDLES cuHandles,
	int Nx, int Ny, int Nz, int Nd, int N, char* lattice_type, realCPU error, PROFILE* Profile)
{
	int i;
	int N2 = N * 2;
	int Ele_field_mtx_N = N * 3;
	int N12 = N * 12;
	size_t size, dsizeEle_field_mtx_N, dsizeNd2;

	size = Ele_field_mtx_N * sizeof(cmpxCPU);

	cmpxCPU* vec_x    = (cmpxCPU*) malloc(size);
	cmpxCPU* vec_y    = (cmpxCPU*) malloc(size);
	cmpxCPU* vec_temp = (cmpxCPU*) malloc(size);

	cmpxGPU* N3_temp1 = cuHandles.N3_temp1;
	cmpxGPU* N3_temp2 = cuHandles.N3_temp2;

	cmpxGPU* Nd2_temp;
	dsizeEle_field_mtx_N = Ele_field_mtx_N * sizeof(cmpxGPU);
	dsizeNd2 = Nd * 2 * sizeof(cmpxGPU);

	checkCudaErrors(cudaMalloc((void**)&Nd2_temp, dsizeNd2));

	srand(time(NULL));

	for(i = 0; i < Ele_field_mtx_N; i++)
	//vec_x[i] = ((realCPU) rand()/(RAND_MAX + 1.0))  for test
		vec_x[i] = ((realCPU) rand()/(RAND_MAX + 1.0)) +  I * ((realCPU) rand()/(RAND_MAX + 1.0));

  realCPU vec_x_norm = vec_norm(vec_x, Ele_field_mtx_N);
  /*for(i = 0; i < Ele_field_mtx_N; i++)
    vec_x[i] = vec_x[i] /vec_x_norm;*/
  
	cmpxCPU *vec_y_1, *vec_y_2, *vec_y_3;

	cudaMemcpy(N3_temp1, vec_x, dsizeEle_field_mtx_N, cudaMemcpyHostToDevice);
  
  //printDeviceArray(N3_temp1, Ele_field_mtx_N, "vec_x.txt");

	if( (strcmp(lattice_type, "simple_cubic"          ) == 0) || \
		(strcmp(lattice_type, "primitive_orthorhombic") == 0) || \
		(strcmp(lattice_type, "primitive_tetragonal"  ) == 0) )
	{
		//FFT_CUDA(cuHandles, fft_buffer, N3_temp1, Nx, Ny, Nz, Lambdas_cuda.dD_ks, N3_temp2);
		FFT_CUDA(N3_temp2, N3_temp1, Lambdas_cuda.dD_ks, fft_buffer, cuHandles, Nx, Ny, Nz);
	}
	else
	{
		for(i = 0; i < 3; i++)
        	spMV_fastT_gpu(N3_temp2+i*N, N3_temp1+i*N, cuHandles, &fft_buffer, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Nx, Ny, Nz, -1);
	}

	cudaMemcpy(vec_y, N3_temp2, dsizeEle_field_mtx_N, cudaMemcpyDeviceToHost);
	vec_y_1 = &vec_y[0];  vec_y_2 = &vec_y[N];  vec_y_3 = &vec_y[N2];

	if(Nd == N - 1)
	{
		vec_y_1[0] = 0; vec_y_2[0] = 0; vec_y_3[0] = 0;
		for(i = 0; i < N - 1; i++)
		{
			vec_y_1[i + 1] = Lambdas.Lambda_x[i] * vec_y_1[i + 1];
			vec_y_2[i + 1] = Lambdas.Lambda_y[i] * vec_y_2[i + 1];
			vec_y_3[i + 1] = Lambdas.Lambda_z[i] * vec_y_3[i + 1];
		}
	}
	else
	{
		for(i = 0; i < N; i++)
		{
			vec_y_1[i] = Lambdas.Lambda_x[i] * vec_y_1[i];
			vec_y_2[i] = Lambdas.Lambda_y[i] * vec_y_2[i];
			vec_y_3[i] = Lambdas.Lambda_z[i] * vec_y_3[i];
		}
	}

	cudaMemcpy(N3_temp1, vec_y, dsizeEle_field_mtx_N, cudaMemcpyHostToDevice);

	if((strcmp(lattice_type, "simple_cubic"          ) == 0) || \
	   (strcmp(lattice_type, "primitive_orthorhombic") == 0) || \
	   (strcmp(lattice_type, "primitive_tetragonal"  ) == 0))
	{
		IFFT_CUDA(N3_temp2, N3_temp1, Lambdas_cuda.dD_k, fft_buffer, cuHandles, Nx, Ny, Nz);
		//IFFT_CUDA(cuHandles, fft_buffer, N3_temp1, Nx, Ny, Nz, Lambdas_cuda.dD_k, N3_temp2);
	}
	else
	{
		for(i = 0; i < 3; i++)
			spMV_fastT_gpu(N3_temp2+i*N, N3_temp1+i*N, cuHandles, &fft_buffer, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Nx, Ny, Nz, 1);
	}

	cudaMemcpy(vec_y, N3_temp2, dsizeEle_field_mtx_N, cudaMemcpyDeviceToHost);

	mtx_prod(&vec_temp[0] , mtx_C.C1_r, mtx_C.C1_c, mtx_C.C1_v, &vec_x[0] , N2, N);
	mtx_prod(&vec_temp[N] , mtx_C.C2_r, mtx_C.C2_c, mtx_C.C2_v, &vec_x[N] , N2, N);
	mtx_prod(&vec_temp[N2], mtx_C.C3_r, mtx_C.C3_c, mtx_C.C3_v, &vec_x[N2], N2, N);

	size = N * sizeof(cmpxCPU);
	cmpxCPU* test_x = (cmpxCPU*) malloc(size);
	cmpxCPU* test_y = (cmpxCPU*) malloc(size);
	cmpxCPU* test_z = (cmpxCPU*) malloc(size);

	vec_plus(test_x, 1.0, &vec_temp[0] , -1.0, &vec_y[0] , N);
	vec_plus(test_y, 1.0, &vec_temp[N] , -1.0, &vec_y[N] , N);
	vec_plus(test_z, 1.0, &vec_temp[N2], -1.0, &vec_y[N2], N);
	
	realCPU C1_error = vec_norm(test_x, N)/sqrt(N);
    realCPU C2_error = vec_norm(test_y, N)/sqrt(N);
    realCPU C3_error = vec_norm(test_z, N)/sqrt(N);

	free(test_x); free(test_y); free(test_z);

	cmpxCPU* Qrs_x = (cmpxCPU*) malloc(2*Nd*sizeof(cmpxCPU));

	cudaMemcpy(N3_temp1, vec_x, dsizeEle_field_mtx_N, cudaMemcpyHostToDevice);

	if((strcmp(lattice_type, "simple_cubic"          ) == 0) || \
	   (strcmp(lattice_type, "primitive_orthorhombic") == 0) || \
	   (strcmp(lattice_type, "primitive_tetragonal"  ) == 0))
	{
		FAME_Matrix_Vector_Production_Qrs(Nd2_temp, N3_temp1, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_ks, Lambdas_cuda.dPi_Qrs);
	}
	else
	{
		FAME_Matrix_Vector_Production_Qrs(Nd2_temp, N3_temp1, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Qrs);
	}
  
//  printDeviceArray(Nd2_temp, 2*Nd, "QRS.txt");

	cudaMemcpy(Qrs_x, Nd2_temp, dsizeNd2, cudaMemcpyDeviceToHost);

	for(i = 0; i < Nd; i++ )
	{
		Qrs_x[i]      = Qrs_x[i]      * Lambdas.Lambda_q_sqrt[i];
		Qrs_x[i + Nd] = Qrs_x[i + Nd] * Lambdas.Lambda_q_sqrt[i];
	}

	cudaMemcpy(Nd2_temp, Qrs_x, dsizeNd2, cudaMemcpyHostToDevice);


	if((strcmp(lattice_type, "simple_cubic"          ) == 0) || \
	   (strcmp(lattice_type, "primitive_orthorhombic") == 0) || \
	   (strcmp(lattice_type, "primitive_tetragonal"  ) == 0))
	{
		FAME_Matrix_Vector_Production_Pr(N3_temp1, Nd2_temp, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_k, Lambdas_cuda.dPi_Pr);
	}
	else
	{
		FAME_Matrix_Vector_Production_Pr(N3_temp1, Nd2_temp, cuHandles, fft_buffer, Nx, Ny, Nz, Nd, Lambdas_cuda.dD_kx, Lambdas_cuda.dD_ky, Lambdas_cuda.dD_kz, Lambdas_cuda.dPi_Pr);
	}
  
//  printDeviceArray(N3_temp1, Ele_field_mtx_N, "PrS.txt");
//getchar();
  
	cudaMemcpy(vec_y, N3_temp1, dsizeEle_field_mtx_N, cudaMemcpyDeviceToHost);

	mtx_prod(vec_temp, mtx_C.C_r, mtx_C.C_c, mtx_C.C_v, vec_x, N12, Ele_field_mtx_N);
	
/*cout<<"vec_temp "<<creal(vec_temp[0] )<<"  "<<cimag(vec_temp[0] )<<endl;
cout<<"vec_temp "<<creal(vec_temp[1] )<<"  "<<cimag(vec_temp[1] )<<endl;
cout<<" vec_x "<<creal( vec_x[0] )<<"  "<<cimag( vec_x[0] )<<endl;
cout<<" vec_x "<<creal( vec_x[1] )<<"  "<<cimag( vec_x[1] )<<endl;*/

	cmpxCPU* test = (cmpxCPU*) malloc(Ele_field_mtx_N * sizeof(cmpxCPU));
	vec_plus(test, 1.0, vec_temp, -1.0, vec_y, Ele_field_mtx_N);
	realCPU SVD_test_C = vec_norm(test, Ele_field_mtx_N)/sqrt(Ele_field_mtx_N);

	printf("          EigDecomp_test_C1 = %e\n", C1_error);
    printf("          EigDecomp_test_C2 = %e\n", C2_error);
    printf("          EigDecomp_test_C3 = %e\n", C3_error);
	printf("          SVD_test_C        = %e\n", SVD_test_C);

  //printDeviceArray(N3_temp1, Ele_field_mtx_N, "vec_y.txt");
//getchar();
	if(C1_error > error || C2_error > error || C3_error > error || SVD_test_C > error)
	{
		printf("\033[40;31mFAME_Main_Code(330):\033[0m\n");
		printf("\033[40;31mThe eigen decomposition is not correct.\033[0m\n");
		printf("\033[40;31mIf N = Nx * Ny * Nz > 256^3, may be caused by numerical errors, please loosen 1e-6.\n");
		printf("\033[40;31mIf not, please contact us.\033[0m\n");
		assert(0);
	}
	
  
	cudaFree(Nd2_temp);
	free(test); free(vec_temp); free(Qrs_x);
	free(vec_x); free(vec_y);
}

void Check_Residual_Iso(realCPU* Freq_array, cmpxCPU* Ele_field_mtx, MTX_B mtx_B, MTX_C mtx_C, int N, int Nwant, realCPU error)
{
	int Ele_field_mtx_N = N * 3;
	int N12 = N * 12;
	size_t size;

	size = Ele_field_mtx_N * Nwant * sizeof(cmpxCPU);

	cmpxCPU* vec_temp = (cmpxCPU*)malloc(size);
	cmpxCPU* vec_left = (cmpxCPU*)malloc(size);
	cmpxCPU* residual = (cmpxCPU*)malloc(size);

	realCPU res, omega2, res_inf;
	realCPU* B_eps = (realCPU*)calloc(Ele_field_mtx_N, sizeof(realCPU));
	checkCudaErrors(cudaMemcpy(B_eps, mtx_B.B_eps, Ele_field_mtx_N*sizeof(realCPU), cudaMemcpyDeviceToHost));

	for(int i = 0; i < Nwant; i++)
	{
		omega2 = -pow(Freq_array[i], 2);
		mtx_prod(vec_temp, mtx_C.C_r, mtx_C.C_c, mtx_C.C_v, Ele_field_mtx + i*Ele_field_mtx_N, N12, Ele_field_mtx_N);
		mtx_prod(vec_left, mtx_C.C_r, mtx_C.C_c, mtx_C.C_v, vec_temp, N12, Ele_field_mtx_N, "Conjugate Transpose");
		mtx_dot_prod(B_eps, Ele_field_mtx + i*Ele_field_mtx_N, vec_temp, Ele_field_mtx_N, 1);
   
   		vec_plus(residual, 1.0, vec_left, omega2, vec_temp, Ele_field_mtx_N);
		res = vec_norm(residual, Ele_field_mtx_N);
 
   		res_inf = cabs(residual[0]);
   		for (int j = 1; j < Ele_field_mtx_N; j++)
   		{
       		if(cabs(residual[j]) > res_inf)
        	res_inf = cabs(residual[j]);
    	}
    
		if(res > error)
		{
			printf("\033[40;31mFreq(%2d) = %10.8f, residual = %e\033[0m ,residual_inf = %e.\033[0m\n", i, Freq_array[i], res, res_inf);
			// Freq_array[i] = -Freq_array[i];
		}
		else
			printf("Freq(%2d) = %10.8f, residual = %e, residual_inf = %e.\033[0m\n", i, Freq_array[i], res, res_inf);
	}	

	free(vec_temp); free(vec_left); free(residual);
	free(B_eps);
}

void Check_Residual_Aniso(realCPU* Freq_array, cmpxCPU* Ele_field_mtx, cmpxCPU* Dis_field_mtx, MTX_B mtx_B, MTX_C mtx_C, int N, int Nwant, realCPU error)
{
	int Ele_field_mtx_N = N * 12;
	int N3 = N * 3;
	int N12 = N * 12;
	size_t size;

	size = Ele_field_mtx_N * sizeof(cmpxCPU);

	cmpxCPU* vec_temp = (cmpxCPU*)malloc(size);
	cmpxCPU* vec_left = (cmpxCPU*)malloc(size);
	cmpxCPU* residual = (cmpxCPU*)malloc(size);

	realCPU res, omega2;

	for(int i = 0; i < Nwant; i++)
	{
		omega2 = -pow(Freq_array[i], 2);
		mtx_prod(vec_temp, mtx_C.C_r, mtx_C.C_c, mtx_C.C_v, Ele_field_mtx + i*Ele_field_mtx_N, N12, N3);
		mtx_prod(vec_temp + N3, mtx_C.C_110_r, mtx_C.C_110_c, mtx_C.C_110_v, Ele_field_mtx + i*Ele_field_mtx_N + N3, N12, N3);
		mtx_prod(vec_temp + 2*N3, mtx_C.C_101_r, mtx_C.C_101_c, mtx_C.C_101_v, Ele_field_mtx + i*Ele_field_mtx_N + 2*N3, N12, N3);
		mtx_prod(vec_temp + 3*N3, mtx_C.C_011_r, mtx_C.C_011_c, mtx_C.C_011_v, Ele_field_mtx + i*Ele_field_mtx_N + 3*N3, N12, N3);
		mtx_prod(vec_left, mtx_C.C_r, mtx_C.C_c, mtx_C.C_v, vec_temp, N12, N3, "Conjugate Transpose");
		mtx_prod(vec_left + N3, mtx_C.C_110_r, mtx_C.C_110_c, mtx_C.C_110_v, vec_temp + N3, N12, N3, "Conjugate Transpose");
		mtx_prod(vec_left + 2*N3, mtx_C.C_101_r, mtx_C.C_101_c, mtx_C.C_101_v, vec_temp + 2*N3, N12, N3, "Conjugate Transpose");
		mtx_prod(vec_left + 3*N3, mtx_C.C_011_r, mtx_C.C_011_c, mtx_C.C_011_v, vec_temp + 3*N3, N12, N3, "Conjugate Transpose");
   
   		vec_plus(residual, 1.0, vec_left, omega2, Dis_field_mtx + i*Ele_field_mtx_N, Ele_field_mtx_N);
		res = vec_norm(residual, Ele_field_mtx_N);

		if(res > error)
		{
			printf("\033[40;31mFreq(%2d) = %10.8f, residual = %e.\033[0m\n", i, Freq_array[i], res);
				//Freq_array[i] = -Freq_array[i];
		}
		else
		{
			printf("Freq(%2d) = %10.8f, residual = %e.\n", i, Freq_array[i], res);
		}

	}

	free(vec_temp); free(vec_left); free(residual);
	
}

void Check_Residual_Biiso(realCPU* Freq_array, cmpxCPU* Ele_field_mtx, MTX_B mtx_B, MTX_C mtx_C, int N, int Nwant, realCPU error)
{
	int mtx_N = N * 6;
	int N3 = N * 3;
	int N12 = N * 12;
	size_t size;
	
	size = mtx_N * Nwant * sizeof(cmpxCPU);
	cmpxCPU scal=0.0+1.0*I;

	cmpxCPU* vec_temp = (cmpxCPU*)malloc(size);
	cmpxCPU* vec_left = (cmpxCPU*)malloc(size);
	cmpxCPU* residual = (cmpxCPU*)malloc(size);

	realCPU res, omega2;
	realCPU* B_eps = (realCPU*)calloc(N3, sizeof(realCPU));
	cmpxCPU* B_zeta = (cmpxCPU*)calloc(N3, sizeof(cmpxCPU));
	realCPU* B_mu = (realCPU*)calloc(N3, sizeof(realCPU));
	cmpxCPU* B_xi = (cmpxCPU*)calloc(N3, sizeof(cmpxCPU));
	checkCudaErrors(cudaMemcpy(B_eps, mtx_B.B_eps, N3*sizeof(realCPU), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(B_zeta, mtx_B.B_zeta, N3*sizeof(cmpxGPU), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(B_mu, mtx_B.B_mu, N3*sizeof(realCPU), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(B_xi, mtx_B.B_xi, N3*sizeof(cmpxGPU), cudaMemcpyDeviceToHost));


	for(int i = 0; i < Nwant; i++)
	{
		omega2 = -Freq_array[i];

		mtx_dot_prod(B_zeta, Ele_field_mtx + i*mtx_N, residual, N3, 1);
		mtx_dot_prod(B_mu, Ele_field_mtx + i*mtx_N + N3, vec_left, N3, 1);
		mtx_dot_prod(B_eps, Ele_field_mtx + i*mtx_N, residual + N3, N3, 1);
		mtx_dot_prod(B_xi, Ele_field_mtx + i*mtx_N + N3, vec_left + N3, N3, 1);


		vec_plus(vec_temp, scal, vec_left, scal, residual, N3);
		vec_plus(vec_temp + N3, -scal, vec_left + N3, -scal, residual + N3, N3);

		mtx_prod(vec_left, mtx_C.C_r, mtx_C.C_c, mtx_C.C_v, Ele_field_mtx + i*mtx_N, N12, N3);
		mtx_prod(vec_left + N3, mtx_C.C_r, mtx_C.C_c, mtx_C.C_v, Ele_field_mtx + i*mtx_N + N3, N12, N3, "Conjugate Transpose");
			
		vec_plus(residual, 1.0, vec_left, omega2, vec_temp, mtx_N);

		res = vec_norm(residual, mtx_N);

		if(res > error)
		{
			printf("\033[40;31mFreq(%2d) = %10.8f, residual = %e.\033[0m\n", i, Freq_array[i], res);
				//Freq_array[i] = -Freq_array[i];
		}
		else
			printf("Freq(%2d) = %10.8f, residual = %e.\n", i, Freq_array[i], res);
	}

	free(vec_temp); free(vec_left); free(residual);
	free(B_eps);free(B_zeta);free(B_xi);free(B_mu);
	
}

void Check_Residual_Bianiso(realCPU* Freq_array, cmpxCPU* Ele_field_mtx, cmpxCPU* Dis_field_mtx, MTX_B mtx_B, MTX_C mtx_C, int N, int Nwant, realCPU error)
{
	int mtx_N = N * 48;
	int N3 = N * 3;
	int N12 = N * 12;
	size_t size;
	
	size = mtx_N * Nwant * sizeof(cmpxCPU);
	cmpxCPU scal = 0.0 + 1.0*I;

	cmpxCPU* vec_left = (cmpxCPU*)malloc(size);
	cmpxCPU* residual = (cmpxCPU*)malloc(size);

	realCPU res;
	cmpxCPU omega2;

	for(int i = 0; i < Nwant; i++)
	{
		omega2 = -Freq_array[i] * scal;

		mtx_prod(vec_left + 0 * N3,  mtx_C.C_r,     mtx_C.C_c,     mtx_C.C_v,     Ele_field_mtx + i * mtx_N + 12 * N3, N12, N3, "minus", "Conjugate Transpose");
		mtx_prod(vec_left + 1 * N3,  mtx_C.C_110_r, mtx_C.C_110_c, mtx_C.C_110_v, Ele_field_mtx + i * mtx_N + 13 * N3, N12, N3, "minus", "Conjugate Transpose");
		mtx_prod(vec_left + 2 * N3,  mtx_C.C_101_r, mtx_C.C_101_c, mtx_C.C_101_v, Ele_field_mtx + i * mtx_N + 14 * N3, N12, N3, "minus", "Conjugate Transpose");
		mtx_prod(vec_left + 3 * N3,  mtx_C.C_011_r, mtx_C.C_011_c, mtx_C.C_011_v, Ele_field_mtx + i * mtx_N + 15 * N3, N12, N3, "minus", "Conjugate Transpose");

		mtx_prod(vec_left + 4 * N3,  mtx_C.C_r,     mtx_C.C_c,     mtx_C.C_v,     Ele_field_mtx + i * mtx_N + 8 * N3, N12, N3, "Conjugate Transpose");
		mtx_prod(vec_left + 5 * N3,  mtx_C.C_110_r, mtx_C.C_110_c, mtx_C.C_110_v, Ele_field_mtx + i * mtx_N + 9 * N3, N12, N3, "Conjugate Transpose");
		mtx_prod(vec_left + 6 * N3,  mtx_C.C_101_r, mtx_C.C_101_c, mtx_C.C_101_v, Ele_field_mtx + i * mtx_N + 10 * N3, N12, N3, "Conjugate Transpose");
		mtx_prod(vec_left + 7 * N3,  mtx_C.C_011_r, mtx_C.C_011_c, mtx_C.C_011_v, Ele_field_mtx + i * mtx_N + 11 * N3, N12, N3, "Conjugate Transpose");

		mtx_prod(vec_left + 8 * N3,  mtx_C.C_r,     mtx_C.C_c,     mtx_C.C_v,     Ele_field_mtx + i * mtx_N + 4 * N3, N12, N3, "minus");
		mtx_prod(vec_left + 9 * N3,  mtx_C.C_110_r, mtx_C.C_110_c, mtx_C.C_110_v, Ele_field_mtx + i * mtx_N + 5 * N3, N12, N3, "minus");
		mtx_prod(vec_left + 10 * N3, mtx_C.C_101_r, mtx_C.C_101_c, mtx_C.C_101_v, Ele_field_mtx + i * mtx_N + 6 * N3, N12, N3, "minus");
		mtx_prod(vec_left + 11 * N3, mtx_C.C_011_r, mtx_C.C_011_c, mtx_C.C_011_v, Ele_field_mtx + i * mtx_N + 7 * N3, N12, N3, "minus");

		mtx_prod(vec_left + 12 * N3, mtx_C.C_r,     mtx_C.C_c,     mtx_C.C_v,     Ele_field_mtx + i * mtx_N + 0 * N3, N12, N3);
		mtx_prod(vec_left + 13 * N3, mtx_C.C_110_r, mtx_C.C_110_c, mtx_C.C_110_v, Ele_field_mtx + i * mtx_N + 1 * N3, N12, N3);
		mtx_prod(vec_left + 14 * N3, mtx_C.C_101_r, mtx_C.C_101_c, mtx_C.C_101_v, Ele_field_mtx + i * mtx_N + 2 * N3, N12, N3);
		mtx_prod(vec_left + 15 * N3, mtx_C.C_011_r, mtx_C.C_011_c, mtx_C.C_011_v, Ele_field_mtx + i * mtx_N + 3 * N3, N12, N3);
			
		vec_plus(residual, 1.0, vec_left, omega2, Dis_field_mtx + i * mtx_N, mtx_N);
		res = vec_norm(residual, mtx_N);

		if(res > error)
		{
			printf("\033[40;31mFreq(%2d) = %10.8f, residual = %e.\033[0m\n", i, Freq_array[i], res);
				//Freq_array[i] = -Freq_array[i];
		}
		else
		{
			printf("Freq(%2d) = %10.8f, residual = %e.\n", i, Freq_array[i], res);
		}

	}

	free(vec_left); free(residual);
	
}

