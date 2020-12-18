#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include "FAME_FFT_CUDA.cuh"

static __global__ void vp_add_vp_add_vp(int N, int Nd, int Nd_2, cmpxGPU* L, cmpxGPU* vec, cmpxGPU* vec_out);
////////////=========================== Create Qrs function for Biiso (cuda)===========================//////////////////
int FAME_Matrix_Vector_Production_Qrs(
	cmpxGPU* vec_y,
	cmpxGPU* vec_x,
	CULIB_HANDLES    cuHandles, 
	FFT_BUFFER       fft_buffer, 
	cmpxGPU* D_ks, 
	cmpxGPU* Pi_Qrs, 
	int Nx, int Ny, int Nz, int Nd, PROFILE* Profile)
{
    int N   = Nx * Ny * Nz;
    int Nd2 = Nd * 2;

    dim3 DimBlock(BLOCK_SIZE,1,1);
    dim3 DimGrid((Nd-1)/BLOCK_SIZE +1,1,1);
//	struct timespec start, end;
//	clock_gettime (CLOCK_REALTIME, &start);
	FFT_CUDA(vec_x, vec_x, D_ks, fft_buffer, cuHandles, Nx, Ny, Nz);
// Time end 
//	clock_gettime (CLOCK_REALTIME, &end);	
//	Profile->fft_time[Profile->idx] += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / BILLION;
	//cout<<"fft "<< (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / BILLION <<endl;

	vp_add_vp_add_vp<<<DimGrid, DimBlock>>>(N, Nd, Nd2, Pi_Qrs,    vec_x+(N-Nd), vec_y);
	vp_add_vp_add_vp<<<DimGrid, DimBlock>>>(N, Nd, Nd2, Pi_Qrs+Nd, vec_x+(N-Nd), vec_y+Nd);

    return 0;
}

int FAME_Matrix_Vector_Production_Qrs(
	cmpxGPU* vec_y,
	cmpxGPU* vec_x,
	CULIB_HANDLES    cuHandles, 
	FFT_BUFFER       fft_buffer, 
	cmpxGPU* D_kx, 
	cmpxGPU* D_ky, 
	cmpxGPU* D_kz, 
	cmpxGPU* Pi_Qrs, 
	int Nx, int Ny, int Nz, int Nd, PROFILE* Profile)
{

    int N   = Nx * Ny * Nz;
    int N2  = N * 2;
    int Nd2 = Nd * 2;
    dim3 DimBlock(BLOCK_SIZE,1,1);
    dim3 DimGrid((N-1)/BLOCK_SIZE +1,1,1);

//	struct timespec start, end;
//	clock_gettime (CLOCK_REALTIME, &start);
    spMV_fastT_gpu(vec_x,    vec_x,    cuHandles, &fft_buffer, D_kx, D_ky, D_kz, Nx, Ny, Nz, -1);
    spMV_fastT_gpu(vec_x+N,  vec_x+N,  cuHandles, &fft_buffer, D_kx, D_ky, D_kz, Nx, Ny, Nz, -1);
    spMV_fastT_gpu(vec_x+N2, vec_x+N2, cuHandles, &fft_buffer, D_kx, D_ky, D_kz, Nx, Ny, Nz, -1);
// Time end 
//	clock_gettime (CLOCK_REALTIME, &end);	
//	Profile->fft_time[Profile->idx] += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / BILLION;
	
	// Pi_Qrs*vec
 	vp_add_vp_add_vp<<<DimGrid, DimBlock>>>(N, Nd, Nd2, Pi_Qrs,    vec_x+(N-Nd), vec_y);
 	vp_add_vp_add_vp<<<DimGrid, DimBlock>>>(N, Nd, Nd2, Pi_Qrs+Nd, vec_x+(N-Nd), vec_y+Nd);

 	return 0;
}

static __global__ void vp_add_vp_add_vp(int N, int Nd, int Nd_2, cmpxGPU* L, cmpxGPU* vec, cmpxGPU* vec_out)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < Nd)
    {
        //vec_out[idx] = L_1[idx]*vec_1[idx] + L_3[idx]*vec_3[idx] + L_2[idx]*vec_2[idx]

        vec_out[idx].x = L[idx].x*vec[idx].x + L[idx+Nd_2].x*vec[idx+N].x + L[idx+2*Nd_2].x*vec[idx+2*N].x\
                         - L[idx].y*vec[idx].y - L[idx+Nd_2].y*vec[idx+N].y - L[idx+2*Nd_2].y*vec[idx+2*N].y;

        vec_out[idx].y = L[idx].x*vec[idx].y + L[idx+Nd_2].x*vec[idx+N].y + L[idx+2*Nd_2].x*vec[idx+2*N].y\
                         + L[idx].y*vec[idx].x + L[idx+Nd_2].y*vec[idx+N].x + L[idx+2*Nd_2].y*vec[idx+2*N].x;

    }

}

