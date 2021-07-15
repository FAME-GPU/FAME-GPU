#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include <complex.h>
#include "printDeviceArray.cuh"

static __global__ void G_vec_prod(cmpxGPU* G_in, 
    cmpxGPU* vec0, cmpxGPU* vec1, cmpxGPU* vec2,
    cmpxGPU* vec3, cmpxGPU* vec4, cmpxGPU* vec5, 
    cmpxGPU* temp_vec0, cmpxGPU* temp_vec1, cmpxGPU* temp_vec2, 
    cmpxGPU* temp_vec3, cmpxGPU* temp_vec4, cmpxGPU* temp_vec5, 
    int* InOut_index, int InOut_index_length);

void FAME_Matrix_Vector_Production_Bianisotropic_dualG_Shuffle(CULIB_HANDLES cuHandles, int Nx, int Ny, int Nz, int Nd, 
    int* InOut_index, int* InOut_index_length, cmpxGPU* G_in, cmpxGPU* vec)
{
    int N = Nx * Ny * Nz;

    cmpxGPU* temp_vec = cuHandles.temp_vec1;
    cudaMemcpy(temp_vec, vec, 24*N*sizeof(cmpxGPU), cudaMemcpyDeviceToDevice);
     
    dim3 DimBlock(256,1,1);
    dim3 DimGrid((N-1)/256 +1,1,1);

    G_vec_prod<<<DimGrid, DimBlock>>>(G_in, temp_vec, temp_vec+4*N, temp_vec+8*N, temp_vec+12*N, temp_vec+16*N, temp_vec+20*N,
        vec, vec+4*N,  vec+8*N, vec+12*N, vec+16*N, vec+20*N, 
        InOut_index+InOut_index_length[3], InOut_index_length[4] - InOut_index_length[3]);
    cudaDeviceSynchronize();

    G_vec_prod<<<DimGrid, DimBlock>>>(G_in, temp_vec+3*N, temp_vec+1*N, temp_vec+11*N, temp_vec+15*N, temp_vec+13*N, temp_vec+23*N,
        vec+3*N, vec+1*N, vec+11*N, vec+15*N, vec+13*N, vec+23*N, 
        InOut_index+InOut_index_length[4], InOut_index_length[4] - InOut_index_length[4]);
    cudaDeviceSynchronize();

    G_vec_prod<<<DimGrid, DimBlock>>>(G_in, temp_vec+6*N, temp_vec+10*N, temp_vec+2*N, temp_vec+18*N, temp_vec+22*N, temp_vec+14*N, 
        vec+6*N, vec+10*N, vec+2*N, vec+18*N, vec+22*N, vec+14*N, 
        InOut_index+InOut_index_length[5], InOut_index_length[6] - InOut_index_length[5]);
    cudaDeviceSynchronize();

    G_vec_prod<<<DimGrid, DimBlock>>>(G_in, temp_vec+9*N, temp_vec+7*N, temp_vec+5*N, temp_vec+21*N, temp_vec+19*N, temp_vec+17*N, 
        vec+9*N, vec+7*N, vec+5*N, vec+21*N, vec+19*N, vec+17*N, 
        InOut_index+InOut_index_length[6], InOut_index_length[7] - InOut_index_length[6]);
    cudaDeviceSynchronize();

}


static __global__ void G_vec_prod(cmpxGPU* G_in,
    cmpxGPU* vec0, cmpxGPU* vec1, cmpxGPU* vec2,
    cmpxGPU* vec3, cmpxGPU* vec4, cmpxGPU* vec5, 
    cmpxGPU* temp_vec0, cmpxGPU* temp_vec1, cmpxGPU* temp_vec2, 
    cmpxGPU* temp_vec3, cmpxGPU* temp_vec4, cmpxGPU* temp_vec5, 
    int* InOut_index, int InOut_index_length)
{  

    __shared__ cmpxGPU G[36];
    for(int i = 0; i < 36; i++)
    {
        G[i].x = G_in[i].x;
        G[i].y = G_in[i].y;
    }

    int InOut_idx;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if( idx < InOut_index_length )
    {     
        InOut_idx = InOut_index[idx];

        temp_vec0[InOut_idx].x  = G[0].x * vec0[InOut_idx].x + G[6].x * vec1[InOut_idx].x + G[12].x * vec2[InOut_idx].x + G[18].x * vec3[InOut_idx].x + G[24].x * vec4[InOut_idx].x + G[30].x * vec5[InOut_idx].x - G[0].y * vec0[InOut_idx].y - G[6].y * vec1[InOut_idx].y - G[12].y * vec2[InOut_idx].y - G[18].y * vec3[InOut_idx].y - G[24].y * vec4[InOut_idx].y - G[30].y * vec5[InOut_idx].y;
        temp_vec0[InOut_idx].y  = G[0].x * vec0[InOut_idx].y + G[6].x * vec1[InOut_idx].y + G[12].x * vec2[InOut_idx].y + G[18].x * vec3[InOut_idx].y + G[24].x * vec4[InOut_idx].y + G[30].x * vec5[InOut_idx].y + G[0].y * vec0[InOut_idx].x + G[6].y * vec1[InOut_idx].x + G[12].y * vec2[InOut_idx].x + G[18].y * vec3[InOut_idx].x + G[24].y * vec4[InOut_idx].x + G[30].y * vec5[InOut_idx].x;

        temp_vec1[InOut_idx].x  = G[1].x * vec0[InOut_idx].x + G[7].x * vec1[InOut_idx].x + G[13].x * vec2[InOut_idx].x + G[19].x * vec3[InOut_idx].x + G[25].x * vec4[InOut_idx].x + G[31].x * vec5[InOut_idx].x - G[1].y * vec0[InOut_idx].y - G[7].y * vec1[InOut_idx].y - G[13].y * vec2[InOut_idx].y - G[19].y * vec3[InOut_idx].y - G[25].y * vec4[InOut_idx].y - G[31].y * vec5[InOut_idx].y;
        temp_vec1[InOut_idx].y  = G[1].x * vec0[InOut_idx].y + G[7].x * vec1[InOut_idx].y + G[13].x * vec2[InOut_idx].y + G[19].x * vec3[InOut_idx].y + G[25].x * vec4[InOut_idx].y + G[31].x * vec5[InOut_idx].y + G[1].y * vec0[InOut_idx].x + G[7].y * vec1[InOut_idx].x + G[13].y * vec2[InOut_idx].x + G[19].y * vec3[InOut_idx].x + G[25].y * vec4[InOut_idx].x + G[31].y * vec5[InOut_idx].x;

        temp_vec2[InOut_idx].x  = G[2].x * vec0[InOut_idx].x + G[8].x * vec1[InOut_idx].x + G[14].x * vec2[InOut_idx].x + G[20].x * vec3[InOut_idx].x + G[26].x * vec4[InOut_idx].x + G[32].x * vec5[InOut_idx].x - G[2].y * vec0[InOut_idx].y - G[8].y * vec1[InOut_idx].y - G[14].y * vec2[InOut_idx].y - G[20].y * vec3[InOut_idx].y - G[26].y * vec4[InOut_idx].y - G[32].y * vec5[InOut_idx].y;
        temp_vec2[InOut_idx].y  = G[2].x * vec0[InOut_idx].y + G[8].x * vec1[InOut_idx].y + G[14].x * vec2[InOut_idx].y + G[20].x * vec3[InOut_idx].y + G[26].x * vec4[InOut_idx].y + G[32].x * vec5[InOut_idx].y + G[2].y * vec0[InOut_idx].x + G[8].y * vec1[InOut_idx].x + G[14].y * vec2[InOut_idx].x + G[20].y * vec3[InOut_idx].x + G[26].y * vec4[InOut_idx].x + G[32].y * vec5[InOut_idx].x;

        temp_vec3[InOut_idx].x  = G[3].x * vec0[InOut_idx].x + G[9].x * vec1[InOut_idx].x + G[15].x * vec2[InOut_idx].x + G[21].x * vec3[InOut_idx].x + G[27].x * vec4[InOut_idx].x + G[33].x * vec5[InOut_idx].x - G[3].y * vec0[InOut_idx].y - G[9].y * vec1[InOut_idx].y - G[15].y * vec2[InOut_idx].y - G[21].y * vec3[InOut_idx].y - G[27].y * vec4[InOut_idx].y - G[33].y * vec5[InOut_idx].y;
        temp_vec3[InOut_idx].y  = G[3].x * vec0[InOut_idx].y + G[9].x * vec1[InOut_idx].y + G[15].x * vec2[InOut_idx].y + G[21].x * vec3[InOut_idx].y + G[27].x * vec4[InOut_idx].y + G[33].x * vec5[InOut_idx].y + G[3].y * vec0[InOut_idx].x + G[9].y * vec1[InOut_idx].x + G[15].y * vec2[InOut_idx].x + G[21].y * vec3[InOut_idx].x + G[27].y * vec4[InOut_idx].x + G[33].y * vec5[InOut_idx].x;

        temp_vec4[InOut_idx].x  = G[4].x * vec0[InOut_idx].x + G[10].x * vec1[InOut_idx].x + G[16].x * vec2[InOut_idx].x + G[22].x * vec3[InOut_idx].x + G[28].x * vec4[InOut_idx].x + G[34].x * vec5[InOut_idx].x - G[4].y * vec0[InOut_idx].y - G[10].y * vec1[InOut_idx].y - G[16].y * vec2[InOut_idx].y - G[22].y * vec3[InOut_idx].y - G[28].y * vec4[InOut_idx].y - G[34].y * vec5[InOut_idx].y;
        temp_vec4[InOut_idx].y  = G[4].x * vec0[InOut_idx].y + G[10].x * vec1[InOut_idx].y + G[16].x * vec2[InOut_idx].y + G[22].x * vec3[InOut_idx].y + G[28].x * vec4[InOut_idx].y + G[34].x * vec5[InOut_idx].y + G[4].y * vec0[InOut_idx].x + G[10].y * vec1[InOut_idx].x + G[16].y * vec2[InOut_idx].x + G[22].y * vec3[InOut_idx].x + G[28].y * vec4[InOut_idx].x + G[34].y * vec5[InOut_idx].x;

        temp_vec5[InOut_idx].x  = G[5].x * vec0[InOut_idx].x + G[11].x * vec1[InOut_idx].x + G[17].x * vec2[InOut_idx].x + G[23].x * vec3[InOut_idx].x + G[29].x * vec4[InOut_idx].x + G[35].x * vec5[InOut_idx].x - G[5].y * vec0[InOut_idx].y - G[11].y * vec1[InOut_idx].y - G[17].y * vec2[InOut_idx].y - G[23].y * vec3[InOut_idx].y - G[29].y * vec4[InOut_idx].y - G[35].y * vec5[InOut_idx].y;
        temp_vec5[InOut_idx].y  = G[5].x * vec0[InOut_idx].y + G[11].x * vec1[InOut_idx].y + G[17].x * vec2[InOut_idx].y + G[23].x * vec3[InOut_idx].y + G[29].x * vec4[InOut_idx].y + G[35].x * vec5[InOut_idx].y + G[5].y * vec0[InOut_idx].x + G[11].y * vec1[InOut_idx].x + G[17].y * vec2[InOut_idx].x + G[23].y * vec3[InOut_idx].x + G[29].y * vec4[InOut_idx].x + G[35].y * vec5[InOut_idx].x;
 

    }
}
