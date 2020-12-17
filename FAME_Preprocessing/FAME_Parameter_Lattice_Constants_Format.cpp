#include "FAME_Internal_Common.h"
// 2020-02-19

int FAME_Parameter_Lattice_Constants_Format(LATTICE* Lattice)
{
    if(strcmp(Lattice->lattice_type, "simple_cubic"       ) == 0 || \
       strcmp(Lattice->lattice_type, "face_centered_cubic") == 0 || \
       strcmp(Lattice->lattice_type, "body_centered_cubic") == 0)
    {
        Lattice->lattice_constant.b     = Lattice->lattice_constant.a;
        Lattice->lattice_constant.c     = Lattice->lattice_constant.a;
        Lattice->lattice_constant.alpha = pi / 2;
        Lattice->lattice_constant.beta  = pi / 2;
        Lattice->lattice_constant.gamma = pi / 2;
    }
    else if(strcmp(Lattice->lattice_type, "hexagonal") == 0)
    {
        Lattice->lattice_constant.b     = Lattice->lattice_constant.a;
        Lattice->lattice_constant.alpha = pi / 2;
        Lattice->lattice_constant.beta  = pi / 2;
        Lattice->lattice_constant.gamma = 2 * pi / 3;
    }
    else if(strcmp(Lattice->lattice_type, "rhombohedral") == 0)
    {
        Lattice->lattice_constant.b     = Lattice->lattice_constant.a;
        Lattice->lattice_constant.c     = Lattice->lattice_constant.a;
        Lattice->lattice_constant.beta  = Lattice->lattice_constant.alpha;
        Lattice->lattice_constant.gamma = Lattice->lattice_constant.alpha;
    }
    else if(strcmp(Lattice->lattice_type, "primitive_tetragonal") == 0 || \
            strcmp(Lattice->lattice_type, "body_centered_tetragonal") == 0)
    {
        Lattice->lattice_constant.b     = Lattice->lattice_constant.a;
        Lattice->lattice_constant.alpha = pi / 2;
        Lattice->lattice_constant.beta  = pi / 2;
        Lattice->lattice_constant.gamma = pi / 2;
    }
    else if(strcmp(Lattice->lattice_type, "primitive_orthorhombic") == 0 || \
            strcmp(Lattice->lattice_type, "a_base_centered_orthorhombic") == 0 || \
            strcmp(Lattice->lattice_type, "c_base_centered_orthorhombic") == 0 || \
            strcmp(Lattice->lattice_type, "face_centered_orthorhombic") == 0 || \
            strcmp(Lattice->lattice_type, "body_centered_orthorhombic") == 0)
    {
        Lattice->lattice_constant.alpha = pi / 2;
        Lattice->lattice_constant.beta  = pi / 2;
        Lattice->lattice_constant.gamma = pi / 2;
    }
    else if(strcmp(Lattice->lattice_type, "primitive_monoclinic") == 0 || \
            strcmp(Lattice->lattice_type, "base_centered_monoclinic") == 0)
    {
        Lattice->lattice_constant.beta  = pi / 2;
        Lattice->lattice_constant.gamma = pi / 2;
    }
    else if(strcmp(Lattice->lattice_type, "triclinic") == 0)
    {

    }
    else
    {
        printf("\033[40;31mFAME_Parameter_Lattice_Constants_Format(60):\033[0m\n");
        printf("\033[40;31mLattice type '%s' is invalid! Please check lattice type is correct in material data.\033[0m\n", Lattice->lattice_type);
        assert(0);
    }

    return 0;
}