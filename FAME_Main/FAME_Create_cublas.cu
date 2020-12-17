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

    cufftErr = cufftPlan1d(&cuHandles->cufft_plan_1d_x, Nx, CUFFT_Z2Z, Ny*Nz);
    assert(cufftErr == CUFFT_SUCCESS);

    cufftErr = cufftPlan1d(&cuHandles->cufft_plan_1d_y, Ny, CUFFT_Z2Z, Nx*Nz);
    assert(cufftErr == CUFFT_SUCCESS);

    cufftErr = cufftPlan1d(&cuHandles->cufft_plan_1d_z, Nz, CUFFT_Z2Z, Nx*Ny);
    assert(cufftErr == CUFFT_SUCCESS);

    cufftErr = cufftPlan3d(&cuHandles->cufft_plan, Nz, Ny, Nx, CUFFT_Z2Z);
    assert(cufftErr == CUFFT_SUCCESS);

    return 0;
}

