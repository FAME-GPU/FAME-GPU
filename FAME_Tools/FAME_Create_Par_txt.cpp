#include "FAME_Internal_Common.h"
#include "FAME_Print.h"
// 2020-02-19

int FAME_Create_Par_txt(PAR Par)
{
    FILE *fp;
    fp = fopen("Data_Par.txt", "w");
    assert(fp != NULL);

    fprintf(fp, "======================== Mesh information ========================\n");
    FAME_Print(Par, (char*)"Grid number", fp);
    FAME_Print(Par, (char*)"Edge length", fp);
    FAME_Print(Par, (char*)"Mesh length", fp);

    fprintf(fp, "\n====================== Lattice information =======================\n");
    FAME_Print(Par, (char*)"Lattice type", fp);
    FAME_Print(Par, (char*)"Lattice constant", fp);
    FAME_Print(Par, (char*)"Permutation", fp);
    FAME_Print(Par, (char*)"Length a", fp);
    FAME_Print(Par, (char*)"Lattice vector a (comp)", fp);
    FAME_Print(Par, (char*)"Lattice vector a (orig)", fp);
    FAME_Print(Par, (char*)"Omega", fp);
    FAME_Print(Par, (char*)"Theta", fp);
    FAME_Print(Par, (char*)"Flag", fp);
    FAME_Print(Par, (char*)"M", fp);
    FAME_Print(Par, (char*)"t1 t2 t3 t4", fp);

    fprintf(fp, "\n================= Reciprocal lattice information =================\n");
    FAME_Print(Par, (char*)"Part number", fp);
    FAME_Print(Par, (char*)"Brillouin zone Path string", fp);
    FAME_Print(Par, (char*)"Wave Vector Number", fp);

    fprintf(fp, "\n====================== Material information ======================\n");
    FAME_Print(Par, (char*)"Material data name", fp);
    FAME_Print(Par, (char*)"Material number", fp);
    FAME_Print(Par, (char*)"Material type", fp);
    FAME_Print(Par, (char*)"Sphere number", fp);
    FAME_Print(Par, (char*)"Sphere centers", fp);
    FAME_Print(Par, (char*)"Sphere radius", fp);
    FAME_Print(Par, (char*)"Cylinder number", fp);
    FAME_Print(Par, (char*)"Cylinder top centers", fp);
    FAME_Print(Par, (char*)"Cylinder bot centers", fp);
    FAME_Print(Par, (char*)"Cylinder radius", fp);
    FAME_Print(Par, (char*)"Permittivity (inner material)", fp);
    FAME_Print(Par, (char*)"Permittivity (outer material)", fp);

    fprintf(fp, "\n======================= Solver information =======================\n");
    FAME_Print(Par, (char*)"Desired eigenpair number", fp);
    FAME_Print(Par, (char*)"Dimension of Krylov subspace for Lanczos", fp);
    FAME_Print(Par, (char*)"Tolerance of Lanczos", fp);
    FAME_Print(Par, (char*)"Maximum restart number of Lanczos", fp);
    FAME_Print(Par, (char*)"Tolerance of linear solver", fp);
    FAME_Print(Par, (char*)"Maximum iteration number of linear solver", fp);

    fprintf(fp, "\n========================== Flag setting ==========================\n");
    FAME_Print(Par, (char*)"Device", fp);
    FAME_Print(Par, (char*)"Printf User Option", fp);
    FAME_Print(Par, (char*)"Printf Parameter", fp);
    FAME_Print(Par, (char*)"Create Parameter", fp);
    FAME_Print(Par, (char*)"Create B_inout", fp);
    FAME_Print(Par, (char*)"Create Wave Vector", fp);
    FAME_Print(Par, (char*)"Save Eigen Vector", fp);
    FAME_Print(Par, (char*)"Grid nums Max", fp);
    FAME_Print(Par, (char*)"Sphere/Cylinder radius adjustment", fp);

    fclose(fp);

    return 0;
}
