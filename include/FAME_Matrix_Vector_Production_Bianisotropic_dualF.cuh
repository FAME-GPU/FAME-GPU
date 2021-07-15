#ifndef _FAME_MATRIX_VECTOR_PRODUCTION_BIANISOTROPIC_DUALF_H_
#define _FAME_MATRIX_VECTOR_PRODUCTION_BIANISOTROPIC_DUALF_H_

void FAME_Matrix_Vector_Production_Bianisotropic_dualF(CULIB_HANDLES cuHandles, FFT_BUFFER fft_buffer, MTX_B mtx_B, 
    int Nx, int Ny, int Nz, int Nd, 
    cmpxGPU* Pi_Pr, cmpxGPU* Pi_Prs, 
    cmpxGPU* Pi_Pr_110, cmpxGPU* Pi_Prs_110,
    cmpxGPU* Pi_Pr_101, cmpxGPU* Pi_Prs_101, 
    cmpxGPU* Pi_Pr_011, cmpxGPU* Pi_Prs_011, 
    cmpxGPU* D_k, cmpxGPU* D_ks, 
    cmpxGPU* x, cmpxGPU* y);


void FAME_Matrix_Vector_Production_Bianisotropic_dualF(CULIB_HANDLES cuHandles, FFT_BUFFER fft_buffer, MTX_B mtx_B,
    int Nx, int Ny, int Nz, int Nd, 
    cmpxGPU* Pi_Pr, cmpxGPU* Pi_Prs,
    cmpxGPU* Pi_Pr_110, cmpxGPU* Pi_Prs_110,
    cmpxGPU* Pi_Pr_101, cmpxGPU* Pi_Prs_101,
    cmpxGPU* Pi_Pr_011, cmpxGPU* Pi_Prs_011,
    cmpxGPU* D_kx, cmpxGPU* D_ky, cmpxGPU* D_kz,  
    cmpxGPU* x, cmpxGPU* y);

#endif