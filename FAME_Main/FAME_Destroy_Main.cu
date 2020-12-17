#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"

int FAME_Destroy_Main(
    CULIB_HANDLES  cuHandles,
    FFT_BUFFER     fft_buffer,
    LANCZOS_BUFFER lBuffer,
    MTX_B          mtx_B,
    MTX_C          mtx_C,
    double*        Freq_array,
    cmpx*          Ele_field_mtx)
{
    ///////////////// Destroy cublas and cufft handles /////////////////
	cublasDestroy(cuHandles.cublas_handle);
	cufftDestroy(cuHandles.cufft_plan_1d_x);
    cufftDestroy(cuHandles.cufft_plan_1d_y);
    cufftDestroy(cuHandles.cufft_plan_1d_z);
	cufftDestroy(cuHandles.cufft_plan);

    // Free FFT Buffer
    cudaFree(fft_buffer.d_A);


    // Free temp
    cudaFree(cuHandles.N3_temp1);
    cudaFree(cuHandles.N3_temp2);
 
    // Free MTX_B
    cudaFree(mtx_B.B_eps);
    cudaFree(mtx_B.invB_eps);

    // Free MTX_C
    free(mtx_C.C1_r); free(mtx_C.C1_c); free(mtx_C.C1_v);
    free(mtx_C.C2_r); free(mtx_C.C2_c); free(mtx_C.C2_v);
    free(mtx_C.C3_r); free(mtx_C.C3_c); free(mtx_C.C3_v);
    free(mtx_C.C_r);  free(mtx_C.C_c);  free(mtx_C.C_v);

    free(Freq_array);
    free(Ele_field_mtx);

    
    // Free LANCZOS Buffer
    free(lBuffer.LT0);
    free(lBuffer.LT1); 
    free(lBuffer.z);
    free(lBuffer.T0);
    free(lBuffer.T1);
    free(lBuffer.T2);  
    cudaFree(lBuffer.dU);   
    cudaFree(lBuffer.dz);

	return 0;
}

