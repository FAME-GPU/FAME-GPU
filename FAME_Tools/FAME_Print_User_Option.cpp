#include "FAME_Internal_Common.h"
#include "FAME_Print.h"
// 2020-02-19

int FAME_Print_User_Option( POPT Popt )
{
    printf("======================== Mesh information ========================\n");
    FAME_Print(Popt, (char*)"Grid number");

    printf("\n================= Reciprocal lattice information =================\n");
    FAME_Print(Popt, (char*)"Part number");

    printf("\n====================== Material information ======================\n");
    FAME_Print(Popt, (char*)"Material data name");
    FAME_Print(Popt, (char*)"Material type");
    FAME_Print(Popt, (char*)"Permittivity (inner material)");
    FAME_Print(Popt, (char*)"Permittivity (outer material)");

    printf("\n======================= Solver information =======================\n");
    FAME_Print(Popt, (char*)"Desired eigenpair number");
    FAME_Print(Popt, (char*)"Dimension of Krylov subspace for Lanczos");
    FAME_Print(Popt, (char*)"Tolerance of Lanczos");
    FAME_Print(Popt, (char*)"Maximum restart number of Lanczos");
    FAME_Print(Popt, (char*)"Tolerance of linear solver");
    FAME_Print(Popt, (char*)"Maximum iteration number of linear solver");

    printf("\n========================== Flag setting ==========================\n");
    FAME_Print(Popt, (char*)"Device");
    FAME_Print(Popt, (char*)"Printf User Option");
    FAME_Print(Popt, (char*)"Printf Parameter");
    FAME_Print(Popt, (char*)"Create Parameter");
    FAME_Print(Popt, (char*)"Create B_inout");
    FAME_Print(Popt, (char*)"Create Wave Vector");
    FAME_Print(Popt, (char*)"Save Eigen Vector");
    FAME_Print(Popt, (char*)"Grid nums Max");
    FAME_Print(Popt, (char*)"Sphere/Cylinder radius adjustment");

    return 0;
}
