#include "FAME_Internal_Common.h"
#include "inv3.h"
#include "mtx_prod.h"
// 2020-02-19

void Max_ind_change(realCPU* length_a1, realCPU* length_a2, realCPU* length_a3, realCPU* vec_a, int* permutation);

int FAME_Parameter_Lattice_Vector(LATTICE* lattice)
{
    int i;

    /* content of function Lattice_vector_generate in MATLAB */
    //  Cubic system
    if(strcmp(lattice->lattice_type, "simple_cubic") == 0)
    {
        realCPU a     = lattice->lattice_constant.a;
        realCPU a1[3] = {a, 0, 0};
        realCPU a2[3] = {0, a, 0};
        realCPU a3[3] = {0, 0, a};
        for(i=0; i<3; i++)
        {
            (lattice->lattice_vec_a)[i] = (a1[i]);
            (lattice->lattice_vec_a)[i+3] = (a2[i]);
            (lattice->lattice_vec_a)[i+6] = (a3[i]);
            (lattice->Permutation)[i] = i+1;
        }
    }
    else if(strcmp(lattice->lattice_type, "face_centered_cubic") == 0)
    {
        realCPU a     = 0.5*lattice->lattice_constant.a;
        realCPU a1[3] = {0, a, a};
        realCPU a2[3] = {a, 0, a};
        realCPU a3[3] = {a, a, 0};

        for(i=0; i<3; i++)
        {
            (lattice->lattice_vec_a)[i]   = (a1[i]);
            (lattice->lattice_vec_a)[i+3] = (a2[i]);
            (lattice->lattice_vec_a)[i+6] = (a3[i]);
            (lattice->Permutation)[i] = i+1;
        }
    }
    else if(strcmp(lattice->lattice_type, "body_centered_cubic") == 0)
    {
        realCPU a     = 0.5*lattice->lattice_constant.a;
        realCPU a1[3] = {-1*a,    a,    a};
        realCPU a2[3] = {   a, -1*a,    a};
        realCPU a3[3] = {   a,    a, -1*a};
        for(i=0; i<3; i++)
        {
            (lattice->lattice_vec_a)[i] = (a1[i]);
            (lattice->lattice_vec_a)[i+3] = (a2[i]);
            (lattice->lattice_vec_a)[i+6] = (a3[i]);
            (lattice->Permutation)[i] = i+1;
        }
    }
    //  Tetragonal system
    else if(strcmp(lattice->lattice_type, "primitive_tetragonal") == 0)
    {
        realCPU a     = lattice->lattice_constant.a;
        realCPU c     = lattice->lattice_constant.c;
        realCPU a1[3] = {a, 0, 0};
        realCPU a2[3] = {0, a, 0};
        realCPU a3[3] = {0, 0, c};
        for(i=0; i<3; i++)
        {
            (lattice->lattice_vec_a)[i] = (a1[i]);
            (lattice->lattice_vec_a)[i+3] = (a2[i]);
            (lattice->lattice_vec_a)[i+6] = (a3[i]);
            (lattice->Permutation)[i] = i+1;
        }
    }
    else if(strcmp(lattice->lattice_type, "body_centered_tetragonal") == 0)
    {
        realCPU a     = 0.5*lattice->lattice_constant.a;
        realCPU c     = 0.5*lattice->lattice_constant.c;
        realCPU a1[3] = {-a,  a,  c};
        realCPU a2[3] = { a, -a,  c};
        realCPU a3[3] = { a,  a, -c};
        for(i=0; i<3; i++)
        {
            (lattice->lattice_vec_a)[i] = (a1[i]);
            (lattice->lattice_vec_a)[i+3] = (a2[i]);
            (lattice->lattice_vec_a)[i+6] = (a3[i]);
            (lattice->Permutation)[i] = i+1;
        }
    }
    // Orthorhombic system
    else if(strcmp(lattice->lattice_type, "primitive_orthorhombic") == 0)
    {
        realCPU a = lattice->lattice_constant.a;
        realCPU b = lattice->lattice_constant.b;
        realCPU c = lattice->lattice_constant.c;
        realCPU a1[3] = {a, 0, 0};
        realCPU a2[3] = {0, b, 0};
        realCPU a3[3] = {0, 0, c};
        for(i=0; i<3; i++)
        {
            (lattice->lattice_vec_a)[i] = (a1[i]);
            (lattice->lattice_vec_a)[i+3] = (a2[i]);
            (lattice->lattice_vec_a)[i+6] = (a3[i]);
            (lattice->Permutation)[i] = i+1;
        }
    }
    else if(strcmp(lattice->lattice_type, "face_centered_orthorhombic") == 0)//no check
    {
        realCPU a = 0.5*lattice->lattice_constant.a;
        realCPU b = 0.5*lattice->lattice_constant.b;
        realCPU c = 0.5*lattice->lattice_constant.c;
        realCPU a1[3] = {0, b, c};
        realCPU a2[3] = {a, 0, c};
        realCPU a3[3] = {a, b, 0};
        for(i=0; i<3; i++)
        {
            (lattice->lattice_vec_a)[i] = (a1[i]);
            (lattice->lattice_vec_a)[i+3] = (a2[i]);
            (lattice->lattice_vec_a)[i+6] = (a3[i]);
            (lattice->Permutation)[i] = i+1;
        }
    }
    else if(strcmp(lattice->lattice_type, "body_centered_orthorhombic") == 0)
    {
        realCPU a = 0.5*lattice->lattice_constant.a;
        realCPU b = 0.5*lattice->lattice_constant.b;
        realCPU c = 0.5*lattice->lattice_constant.c;
        realCPU a1[3] = {-a,  b,  c};
        realCPU a2[3] = { a, -b,  c};
        realCPU a3[3] = { a,  b, -c};
        for(i=0; i<3; i++)
        {
            (lattice->lattice_vec_a)[i] = (a1[i]);
            (lattice->lattice_vec_a)[i+3] = (a2[i]);
            (lattice->lattice_vec_a)[i+6] = (a3[i]);
            (lattice->Permutation)[i] = i+1;
        }
    }
    else if(strcmp(lattice->lattice_type, "a_base_centered_orthorhombic") == 0)
    {
        realCPU a = lattice->lattice_constant.a;
        realCPU b = lattice->lattice_constant.b;
        realCPU c = lattice->lattice_constant.c;
        realCPU a1[3] = {a,     0,      0};
        realCPU a2[3] = {0, 0.5*b, -0.5*c};
        realCPU a3[3] = {0, 0.5*b,  0.5*c};
        for(i=0; i<3; i++)
        {
            (lattice->lattice_vec_a)[i] = (a1[i]);
            (lattice->lattice_vec_a)[i+3] = (a2[i]);
            (lattice->lattice_vec_a)[i+6] = (a3[i]);
            (lattice->Permutation)[i] = i+1;
        }
    }
    else if(strcmp(lattice->lattice_type, "c_base_centered_orthorhombic") == 0)
    {
        realCPU a = lattice->lattice_constant.a;
        realCPU b = lattice->lattice_constant.b;
        realCPU c = lattice->lattice_constant.c;
        realCPU a1[3] = {0.5*a, -0.5*b, 0};
        realCPU a2[3] = {0.5*a,  0.5*b, 0};
        realCPU a3[3] = {    0,      0, c};
        for(i=0; i<3; i++)
        {
            (lattice->lattice_vec_a)[i] = (a1[i]);
            (lattice->lattice_vec_a)[i+3] = (a2[i]);
            (lattice->lattice_vec_a)[i+6] = (a3[i]);
            (lattice->Permutation)[i] = i+1;
        }
    }
    //  Hexagonal system
    else if(strcmp(lattice->lattice_type, "hexagonal") == 0)
    {
        realCPU a = lattice->lattice_constant.a;
        realCPU c = lattice->lattice_constant.c;
        realCPU a1[3] = {     a,             0, 0};
        realCPU a2[3] = {-0.5*a, sqrt(3)*0.5*a, 0};
        realCPU a3[3] = {     0,             0, c};
        for(i=0; i<3; i++)
        {
            (lattice->lattice_vec_a)[i] = (a1[i]);
            (lattice->lattice_vec_a)[i+3] = (a2[i]);
            (lattice->lattice_vec_a)[i+6] = (a3[i]);
            (lattice->Permutation)[i] = i+1;
        }
    }
    //  Rhombohedral system
    else if(strcmp(lattice->lattice_type, "rhombohedral") == 0)
    {
        realCPU a = lattice->lattice_constant.a;
        realCPU alpha = lattice->lattice_constant.alpha;
        realCPU a1[3] = {            a*cos(alpha/2), -a*sin(alpha/2),                                                     0};
        realCPU a2[3] = {            a*cos(alpha/2),  a*sin(alpha/2),                                                     0};
        realCPU a3[3] = { a*cos(alpha)/cos(alpha/2),               0, a*sqrt(1.0-(pow(cos(alpha) / cos(alpha / 2), 2)))};
        for(i=0; i<3; i++)
        {
            (lattice->lattice_vec_a)[i] = (a1[i]);
            (lattice->lattice_vec_a)[i+3] = (a2[i]);
            (lattice->lattice_vec_a)[i+6] = (a3[i]);
            (lattice->Permutation)[i] = i+1;
        }
    }
    //  Monoclinic system
    else if(strcmp(lattice->lattice_type, "primitive_monoclinic") == 0)
    {
        realCPU a = lattice->lattice_constant.a;
        realCPU b = lattice->lattice_constant.b;
        realCPU c = lattice->lattice_constant.c;
        realCPU alpha = lattice->lattice_constant.alpha;
        realCPU a1[3] = {a, 0, 0};
        realCPU a2[3] = {0, b, 0};
        realCPU a3[3] = {0, c*cos(alpha), c*sin(alpha)};
        for(i=0; i<3; i++)
        {
            (lattice->lattice_vec_a)[i] = (a1[i]);
            (lattice->lattice_vec_a)[i+3] = (a2[i]);
            (lattice->lattice_vec_a)[i+6] = (a3[i]);
            (lattice->Permutation)[i] = i+1;
        }
    }
    else if(strcmp(lattice->lattice_type, "base_centered_monoclinic") == 0)
    {
        realCPU a = 0.5*lattice->lattice_constant.a;
        realCPU b = 0.5*lattice->lattice_constant.b;
        realCPU c = lattice->lattice_constant.c;
        realCPU alpha = lattice->lattice_constant.alpha;
        realCPU a1[3] = {a, b, 0};
        realCPU a2[3] = {-a, b, 0};
        realCPU a3[3] = {0, c*cos(alpha), c*sin(alpha)};
        for(i=0; i<3; i++)
        {
            (lattice->lattice_vec_a)[i] = (a1[i]);
            (lattice->lattice_vec_a)[i+3] = (a2[i]);
            (lattice->lattice_vec_a)[i+6] = (a3[i]);
            (lattice->Permutation)[i] = i+1;
        }
    }
    //  Triclinic system
    else if(strcmp(lattice->lattice_type, "triclinic") == 0)
    {
        realCPU a = lattice->lattice_constant.a;
        realCPU b = lattice->lattice_constant.b;
        realCPU c = lattice->lattice_constant.c;
        realCPU alpha = lattice->lattice_constant.alpha;
        realCPU beta  = lattice->lattice_constant.beta;
        realCPU gamma = lattice->lattice_constant.gamma;
        realCPU a1[3] = {a, 0, 0};
        realCPU a2[3] = {b*cos(gamma), b*sin(gamma), 0};
        realCPU a3[3] = { c*cos(beta), c*(cos(alpha)-cos(beta)*cos(gamma))/sin(gamma), c*( sqrt(1-pow(cos(alpha), 2)-pow(cos(beta), 2)-pow(cos(gamma), 2)+2*cos(alpha)*cos(beta)*cos(gamma)) / sin(gamma))};
        for(i=0; i<3; i++)
        {
            (lattice->lattice_vec_a)[i] = (a1[i]);
            (lattice->lattice_vec_a)[i+3] = (a2[i]);
            (lattice->lattice_vec_a)[i+6] = (a3[i]);
            (lattice->Permutation)[i] = i+1;
        }
    }
    else
    {
        /*
        printf("\033[40;31mFAME_Parameter_Lattice_Vector(309):\033[0m\n");
        printf("\033[40;31mLattice type %s is invalid! Please check lattice type is correct in material data.\033[0m\n", lattice->lattice_type);
        assert(0);
        */
    }

    /* the content of function FAME_Parameter_Lattice_Vector in MATLAB */

    // initialize
    (lattice->lattice_constant.length_a1) = 0.0;
    (lattice->lattice_constant.length_a2) = 0.0;
    (lattice->lattice_constant.length_a3) = 0.0;

    for(i = 0; i < 9; i++)
        lattice->lattice_vec_a_orig[i] = lattice->lattice_vec_a[i];
    // take norm of a1, a2, a3
    for(i = 0; i < 3; i++)
    {
        (lattice->lattice_constant.length_a1) += pow((lattice->lattice_vec_a)[i], 2);
        (lattice->lattice_constant.length_a2) += pow((lattice->lattice_vec_a)[i+3], 2);
        (lattice->lattice_constant.length_a3) += pow((lattice->lattice_vec_a)[i+6], 2);
    }

    (lattice->lattice_constant.length_a1) = sqrt(lattice->lattice_constant.length_a1);
    (lattice->lattice_constant.length_a2) = sqrt(lattice->lattice_constant.length_a2);
    (lattice->lattice_constant.length_a3) = sqrt(lattice->lattice_constant.length_a3);

////////////////////////////////////////////////////////////////////////////////////////////
    /*  Condition 1 : norm(a1) = max(norm(a1),norm(a2),norm(a3)) */
    if(strcmp(lattice->lattice_type, "simple_cubic") != 0 &&
       strcmp(lattice->lattice_type, "primitive_orthorhombic") != 0 &&
       strcmp(lattice->lattice_type, "primitive_tetragonal") != 0 )
        Max_ind_change(&(lattice->lattice_constant.length_a1), \
                       &(lattice->lattice_constant.length_a2), \
                       &(lattice->lattice_constant.length_a3), \
                        (lattice->lattice_vec_a), \
                        (lattice->Permutation));

    (lattice->lattice_constant.theta_1) = 0.0;
    (lattice->lattice_constant.theta_2) = 0.0;
    (lattice->lattice_constant.theta_3) = 0.0;

    int k;
    for(k=0; k<3; k++)
    {
        (lattice->lattice_constant.theta_1) += (lattice->lattice_vec_a)[k+3]*(lattice->lattice_vec_a)[k+6]/((lattice->lattice_constant.length_a2)*(lattice->lattice_constant.length_a3));
        (lattice->lattice_constant.theta_2) += (lattice->lattice_vec_a)[k]*(lattice->lattice_vec_a)[k+6]/((lattice->lattice_constant.length_a1)*(lattice->lattice_constant.length_a3));
        (lattice->lattice_constant.theta_3) += (lattice->lattice_vec_a)[k]*(lattice->lattice_vec_a)[k+3]/((lattice->lattice_constant.length_a1)*(lattice->lattice_constant.length_a2));

    }
    (lattice->lattice_constant.theta_1) = acos((lattice->lattice_constant.theta_1));
    (lattice->lattice_constant.theta_2) = acos((lattice->lattice_constant.theta_2));
    (lattice->lattice_constant.theta_3) = acos((lattice->lattice_constant.theta_3));

    /* Condition 2 : */
    if(lattice->lattice_constant.length_a2*sin(lattice->lattice_constant.theta_3) <= ((lattice->lattice_constant.length_a3)/sin(lattice->lattice_constant.theta_3))*fabs(cos(lattice->lattice_constant.theta_1)-cos(lattice->lattice_constant.theta_2)*cos(lattice->lattice_constant.theta_3)))
    {
        realCPU temp4;
        // change vector
        for(k=0; k<3; k++)
        {
            temp4 = (lattice->lattice_vec_a)[k+3];
            (lattice->lattice_vec_a)[k+3] = (lattice->lattice_vec_a)[k+6];
            (lattice->lattice_vec_a)[k+6] = temp4;
        }
        temp4 = lattice->Permutation[1];
        lattice->Permutation[1] = lattice->Permutation[2];
        lattice->Permutation[2] = temp4;

        // change length
        temp4 = (lattice->lattice_constant.length_a2);
        (lattice->lattice_constant.length_a2) = (lattice->lattice_constant.length_a3);
        (lattice->lattice_constant.length_a3) = temp4;

        // compute angle
        for(k=0; k<3; k++)
        {
            lattice->lattice_constant.theta_1 = acos(lattice->lattice_vec_a[k+3]*(lattice->lattice_vec_a)[k+6])/((lattice->lattice_constant.length_a2)*(lattice->lattice_constant.length_a3));
            lattice->lattice_constant.theta_2 = acos(lattice->lattice_vec_a[k]*(lattice->lattice_vec_a)[k+6])/((lattice->lattice_constant.length_a1)*(lattice->lattice_constant.length_a3));
            lattice->lattice_constant.theta_3 = acos(lattice->lattice_vec_a[k]*(lattice->lattice_vec_a)[k+3])/((lattice->lattice_constant.length_a1)*(lattice->lattice_constant.length_a2));
        }
    }
    /* Test condition */
    if(strcmp(lattice->lattice_type, "base_centered_monoclinic") == 0)
    {
        if((lattice->lattice_constant.theta_1)>= pi/2)
            (lattice->lattice_constant.theta_1) = pi - (lattice->lattice_constant.theta_1);

        if((lattice->lattice_constant.theta_2)>= pi/2)
            (lattice->lattice_constant.theta_2) = pi - (lattice->lattice_constant.theta_2);

        if((lattice->lattice_constant.theta_3)>= pi/2)
            (lattice->lattice_constant.theta_3) = pi - (lattice->lattice_constant.theta_3);
    }

    /* Final check */
    if(strcmp(lattice->lattice_type, "simple_cubic") != 0 &&
       strcmp(lattice->lattice_type, "primitive_orthorhombic") != 0 &&
       strcmp(lattice->lattice_type, "primitive_tetragonal") != 0 )
        if((lattice->lattice_constant.length_a2) > (lattice->lattice_constant.length_a1)|| \
                (lattice->lattice_constant.length_a3) > (lattice->lattice_constant.length_a1)|| \
                (lattice->lattice_constant.length_a2)*sin(lattice->lattice_constant.theta_3) <= ((lattice->lattice_constant.length_a3)/sin(lattice->lattice_constant.theta_3))*fabs(cos(lattice->lattice_constant.theta_1)-cos(lattice->lattice_constant.theta_2))*cos(lattice->lattice_constant.theta_3)
          )
        {
            printf("\033[40;31mFAME_Parameter_Lattice_Vector(358):\033[0m\n");
            printf("\033[40;31mThe lattice constants does not suitable for computation. Please contact us.\033[0m\n");
            assert(0);
        }

    lattice->lattice_vec_a[0] = lattice->lattice_constant.length_a1;
    lattice->lattice_vec_a[1] = 0.0;
    lattice->lattice_vec_a[2] = 0.0;
    lattice->lattice_vec_a[3] = lattice->lattice_constant.length_a2*cos(lattice->lattice_constant.theta_3);
    lattice->lattice_vec_a[4] = lattice->lattice_constant.length_a2*sin(lattice->lattice_constant.theta_3);
    lattice->lattice_vec_a[5] = 0.0;
    lattice->lattice_vec_a[6] = lattice->lattice_constant.length_a3*cos(lattice->lattice_constant.theta_2);
    lattice->lattice_vec_a[7] = lattice->lattice_constant.length_a3*(cos(lattice->lattice_constant.theta_1) - cos(lattice->lattice_constant.theta_3)*cos(lattice->lattice_constant.theta_2))/sin(lattice->lattice_constant.theta_3);
    lattice->lattice_vec_a[8] = lattice->lattice_constant.length_a3*sqrt(1 - pow(cos(lattice->lattice_constant.theta_3), 2)-pow(cos(lattice->lattice_constant.theta_2), 2)-pow(cos(lattice->lattice_constant.theta_1), 2) \
                                  + 2*cos(lattice->lattice_constant.theta_3)*cos(lattice->lattice_constant.theta_2)*cos(lattice->lattice_constant.theta_1))/sin(lattice->lattice_constant.theta_3);

    realCPU lattice_vec_a_orig_P[9];
    realCPU inv_lattice_vec_a_orig_P[9];
    lattice_vec_a_orig_P[0] = lattice->lattice_vec_a_orig[(lattice->Permutation[0] - 1) * 3 + 0];
    lattice_vec_a_orig_P[1] = lattice->lattice_vec_a_orig[(lattice->Permutation[0] - 1) * 3 + 1];
    lattice_vec_a_orig_P[2] = lattice->lattice_vec_a_orig[(lattice->Permutation[0] - 1) * 3 + 2];
    lattice_vec_a_orig_P[3] = lattice->lattice_vec_a_orig[(lattice->Permutation[1] - 1) * 3 + 0];
    lattice_vec_a_orig_P[4] = lattice->lattice_vec_a_orig[(lattice->Permutation[1] - 1) * 3 + 1];
    lattice_vec_a_orig_P[5] = lattice->lattice_vec_a_orig[(lattice->Permutation[1] - 1) * 3 + 2];
    lattice_vec_a_orig_P[6] = lattice->lattice_vec_a_orig[(lattice->Permutation[2] - 1) * 3 + 0];
    lattice_vec_a_orig_P[7] = lattice->lattice_vec_a_orig[(lattice->Permutation[2] - 1) * 3 + 1];
    lattice_vec_a_orig_P[8] = lattice->lattice_vec_a_orig[(lattice->Permutation[2] - 1) * 3 + 2];
    inv3(lattice_vec_a_orig_P, inv_lattice_vec_a_orig_P);
    mtx_prod(lattice->Omega, lattice->lattice_vec_a, inv_lattice_vec_a_orig_P, 3, 3, 3);

    return 0;
}

void Max_ind_change(realCPU* length_a1, realCPU* length_a2, realCPU* length_a3, realCPU* vec_a, int* permutation)
{
    int i;
    int max_idx;
    int itemp;
    realCPU dtemp;
    if(length_a1[0] >= length_a2[0] && length_a1[0] >= length_a3[0])
    {
        max_idx = 0;
    }
    else if(length_a2[0] >= length_a1[0] && length_a2[0] >= length_a3[0])
    {
        max_idx = 1;
        // change vector
        for(i = 0; i < 3; i++)
        {
            dtemp        = vec_a[i];
            vec_a[i]     = vec_a[i + 3];
            vec_a[i + 3] = dtemp;
        }

        // change permutation
        itemp                = permutation[max_idx];
        permutation[max_idx] = permutation[0];
        permutation[0]       = itemp;

        // change length
        dtemp        = length_a1[0];
        length_a1[0] = length_a2[0];
        length_a2[0] = dtemp;
    }
    else if(length_a3[0] >= length_a1[0] && length_a3[0] >= length_a2[0])
    {
        max_idx = 2;
        // change vector
        for(i = 0; i < 3; i++)
        {
            dtemp      = vec_a[i];
            vec_a[i]   = vec_a[i + 6];
            vec_a[i + 6] = dtemp;
        }

        // change permutation
        itemp                = permutation[max_idx];
        permutation[max_idx] = permutation[0];
        permutation[0]       = itemp;

        // change length
        dtemp        = length_a1[0];
        length_a1[0] = length_a3[0];
        length_a3[0] = dtemp;
    }
}