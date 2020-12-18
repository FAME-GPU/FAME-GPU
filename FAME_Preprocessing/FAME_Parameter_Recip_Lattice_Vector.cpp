#include "FAME_Internal_Common.h"
#include "inv3.h"

int FAME_Parameter_Recip_Lattice_Vector(realCPU* reciprocal_lattice_vector_b, realCPU* lattice_vec_a)
{
    inv3_Trans(lattice_vec_a, reciprocal_lattice_vector_b);
    return 0;
}
