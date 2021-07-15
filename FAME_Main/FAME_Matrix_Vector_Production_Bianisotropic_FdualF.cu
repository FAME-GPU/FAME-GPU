#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"
#include <complex.h>

#include "FAME_Matrix_Vector_Production_Bianisotropic_F.cuh"
#include "FAME_Matrix_Vector_Production_Bianisotropic_dualF.cuh"

#include "printDeviceArray.cuh"


void FAME_Matrix_Vector_Production_Bianisotropic_FdualF(CULIB_HANDLES cuHandles, FFT_BUFFER fft_buffer, MTX_B mtx_B, 
         int Nx, int Ny, int Nz, int Nd, 
         cmpxGPU* Pi_Qr, cmpxGPU* Pi_Pr, cmpxGPU* Pi_Qrs, cmpxGPU* Pi_Prs,
         cmpxGPU* Pi_Qr_110, cmpxGPU* Pi_Pr_110, cmpxGPU* Pi_Qrs_110, cmpxGPU* Pi_Prs_110,
         cmpxGPU* Pi_Qr_101, cmpxGPU* Pi_Pr_101, cmpxGPU* Pi_Qrs_101, cmpxGPU* Pi_Prs_101,
         cmpxGPU* Pi_Qr_011, cmpxGPU* Pi_Pr_011, cmpxGPU* Pi_Qrs_011, cmpxGPU* Pi_Prs_011,
         cmpxGPU* D_k, cmpxGPU* D_ks, 
         cmpxGPU* x, cmpxGPU* y)
{
    
    FAME_Matrix_Vector_Production_Bianisotropic_F(cuHandles, fft_buffer, mtx_B, Nx, Ny, Nz, Nd, 
        Pi_Qr, Pi_Qrs, Pi_Qr_110, Pi_Qrs_110, Pi_Qr_101, Pi_Qrs_101, Pi_Qr_011, Pi_Qrs_011, D_k, D_ks, x, y);
    FAME_Matrix_Vector_Production_Bianisotropic_dualF(cuHandles, fft_buffer, mtx_B, Nx, Ny, Nz, Nd, 
        Pi_Pr, Pi_Prs, Pi_Pr_110, Pi_Prs_110, Pi_Pr_101, Pi_Prs_101, Pi_Pr_011, Pi_Prs_011, D_k, D_ks, x + 16 * Nd, y + 16 * Nd);

}


void FAME_Matrix_Vector_Production_Bianisotropic_FdualF(CULIB_HANDLES cuHandles, FFT_BUFFER fft_buffer, MTX_B mtx_B, 
         int Nx, int Ny, int Nz, int Nd, 
         cmpxGPU* Pi_Qr, cmpxGPU* Pi_Pr, cmpxGPU* Pi_Qrs, cmpxGPU* Pi_Prs,
         cmpxGPU* Pi_Qr_110, cmpxGPU* Pi_Pr_110, cmpxGPU* Pi_Qrs_110, cmpxGPU* Pi_Prs_110,
         cmpxGPU* Pi_Qr_101, cmpxGPU* Pi_Pr_101, cmpxGPU* Pi_Qrs_101, cmpxGPU* Pi_Prs_101,
         cmpxGPU* Pi_Qr_011, cmpxGPU* Pi_Pr_011, cmpxGPU* Pi_Qrs_011, cmpxGPU* Pi_Prs_011,
         cmpxGPU* D_kx, cmpxGPU* D_ky, cmpxGPU* D_kz, 
         cmpxGPU* x, cmpxGPU* y)
{
    
    FAME_Matrix_Vector_Production_Bianisotropic_F(cuHandles, fft_buffer, mtx_B, Nx, Ny, Nz, Nd,
        Pi_Qr, Pi_Qrs, Pi_Qr_110, Pi_Qrs_110, Pi_Qr_101, Pi_Qrs_101, Pi_Qr_011, Pi_Qrs_011, D_kx, D_ky, D_kz, x, y);
    FAME_Matrix_Vector_Production_Bianisotropic_dualF(cuHandles, fft_buffer, mtx_B, Nx, Ny, Nz, Nd, 
        Pi_Pr, Pi_Prs, Pi_Pr_110, Pi_Prs_110, Pi_Pr_101, Pi_Prs_101, Pi_Pr_011, Pi_Prs_011, D_kx, D_ky, D_kz, x + 16 * Nd, y + 16 * Nd);
        
}