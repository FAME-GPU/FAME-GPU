#include "FAME_Internal_Common.h"
#include "FAME_Print.h"
// 2020-02-19

int FAME_Print_Parameter( PAR Par )
{
    printf("======================== Mesh information ========================\n");
    FAME_Print(Par, (char*)"Grid number");
    FAME_Print(Par, (char*)"Edge length");
    FAME_Print(Par, (char*)"Mesh length");

    printf("\n====================== Lattice information =======================\n");
    FAME_Print(Par, (char*)"Lattice type");
    FAME_Print(Par, (char*)"Lattice constant");
    FAME_Print(Par, (char*)"Permutation");
    FAME_Print(Par, (char*)"Length a");
    FAME_Print(Par, (char*)"Lattice vector a (comp)");
    FAME_Print(Par, (char*)"Lattice vector a (orig)");
    FAME_Print(Par, (char*)"Omega");
    FAME_Print(Par, (char*)"Theta");
    FAME_Print(Par, (char*)"Flag");
    FAME_Print(Par, (char*)"M");
    FAME_Print(Par, (char*)"t1 t2 t3 t4");

    printf("\n================= Reciprocal lattice information =================\n");
    FAME_Print(Par, (char*)"Part number");
    FAME_Print(Par, (char*)"Brillouin zone Path string");
    FAME_Print(Par, (char*)"Wave Vector Number");

    printf("\n====================== Material information ======================\n");
    FAME_Print(Par, (char*)"Material data name");
    FAME_Print(Par, (char*)"Material number");
    FAME_Print(Par, (char*)"Material type");
    FAME_Print(Par, (char*)"Sphere number");
    FAME_Print(Par, (char*)"Sphere centers");
    FAME_Print(Par, (char*)"Sphere radius");
    FAME_Print(Par, (char*)"Cylinder number");
    FAME_Print(Par, (char*)"Cylinder top centers");
    FAME_Print(Par, (char*)"Cylinder bot centers");
    FAME_Print(Par, (char*)"Cylinder radius");
    FAME_Print(Par, (char*)"Permittivity (inner material)");
    FAME_Print(Par, (char*)"Permittivity (outer material)");
    FAME_Print(Par, (char*)"Permeability (inner material)");
    FAME_Print(Par, (char*)"Permeability (outer material)");
    FAME_Print(Par, (char*)"Reciprocity (inner material) (must given when biisotropic)");
    FAME_Print(Par, (char*)"Reciprocity (outer material) (must given when biisotropic)");   
    FAME_Print(Par, (char*)"Chirality (inner material) (must given when biisotropic)");
    FAME_Print(Par, (char*)"Chirality (outer material) (must given when biisotropic)");

    printf("\n======================= Solver information =======================\n");
    FAME_Print(Par, (char*)"Desired eigenpair number");
    FAME_Print(Par, (char*)"Dimension of Krylov subspace for Lanczos");
    FAME_Print(Par, (char*)"Tolerance of Lanczos");
    FAME_Print(Par, (char*)"Maximum restart number of Lanczos");
    FAME_Print(Par, (char*)"Tolerance of linear solver");
    FAME_Print(Par, (char*)"Maximum iteration number of linear solver");

    printf("\n========================== Flag setting ==========================\n");
    FAME_Print(Par, (char*)"Device");
    FAME_Print(Par, (char*)"Printf User Option");
    FAME_Print(Par, (char*)"Printf Parameter");
    FAME_Print(Par, (char*)"Create Parameter");
    FAME_Print(Par, (char*)"Create B_inout");
    FAME_Print(Par, (char*)"Create Wave Vector");
    FAME_Print(Par, (char*)"Save Eigen Vector");
    FAME_Print(Par, (char*)"Grid nums Max");
    FAME_Print(Par, (char*)"Sphere/Cylinder radius adjustment");

    return 0;
}
