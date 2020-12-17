#include "FAME_Internal_Common.h"
// 2020-02-19

int FAME_Parameter_Recip_Lattice_Vector(double* reciprocal_lattice_vector_b, double* lattice_vec_a)
{
    inv3_Trans(lattice_vec_a, reciprocal_lattice_vector_b);
    return 0;
}
