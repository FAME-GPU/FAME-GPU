#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"

int FAME_Create_cublas(CULIB_HANDLES* cuHandles, int Nx, int Ny, int Nz)
{
    
    cublasStatus_t cublasErr;
    cufftResult    cufftErr;

    cublasErr = cublasCreate(&cuHandles->cublas_handle);
    assert(cublasErr == CUBLAS_STATUS_SUCCESS);

    cublasErr = cublasSetPointerMode(cuHandles->cublas_handle, CUBLAS_POINTER_MODE_HOST);
    assert(cublasErr == CUBLAS_STATUS_SUCCESS);

    cufftErr = cufftPlan1d(&cuHandles->cufft_plan_1d_x, Nx, PC_cufft_type, Ny*Nz);
    assert(cufftErr == CUFFT_SUCCESS);

    cufftErr = cufftPlan1d(&cuHandles->cufft_plan_1d_y, Ny, PC_cufft_type, Nx*Nz);
    assert(cufftErr == CUFFT_SUCCESS);

    cufftErr = cufftPlan1d(&cuHandles->cufft_plan_1d_z, Nz, PC_cufft_type, Nx*Ny);
    assert(cufftErr == CUFFT_SUCCESS);

    cufftErr = cufftPlan3d(&cuHandles->cufft_plan, Nz, Ny, Nx, PC_cufft_type);
    assert(cufftErr == CUFFT_SUCCESS);

    return 0;
}

