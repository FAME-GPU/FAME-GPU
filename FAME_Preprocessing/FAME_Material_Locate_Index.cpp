#include "FAME_Internal_Common.h"
#include "FAME_Material_Handle.h"
#include "FAME_Matrix_Grid.h"
#include "inv3.h"
#include "vec_sum.h"
// 2020-02-19

int FAME_Material_Locate(realCPU* point_set, realCPU* Omega, realCPU* lattice_vec_a_orig_P, realCPU* invAP, realCPU* lattice_vec_a_orig, PAR Par)
{
    int in_k_material;
    realCPU point_set_orig[3], coef[3];
    realCPU shift_1, shift_2, shift_3;

    /* point_set_orig = point_set * Omega */
    point_set_orig[0] = point_set[0] * Omega[0] + point_set[1] * Omega[1] + point_set[2] * Omega[2];
    point_set_orig[1] = point_set[0] * Omega[3] + point_set[1] * Omega[4] + point_set[2] * Omega[5];
    point_set_orig[2] = point_set[0] * Omega[6] + point_set[1] * Omega[7] + point_set[2] * Omega[8];

    /* coef = point_set_orig * invAP' */
    coef[0] = point_set_orig[0] * invAP[0] + point_set_orig[1] * invAP[3] + point_set_orig[2] * invAP[6];
    coef[1] = point_set_orig[0] * invAP[1] + point_set_orig[1] * invAP[4] + point_set_orig[2] * invAP[7];
    coef[2] = point_set_orig[0] * invAP[2] + point_set_orig[1] * invAP[5] + point_set_orig[2] * invAP[8];

    shift_1 = floor(coef[0]);
    shift_2 = floor(coef[1]);
    shift_3 = floor(coef[2]);

    point_set_orig[0] = point_set_orig[0] - (shift_1 * lattice_vec_a_orig_P[0] + shift_2 * lattice_vec_a_orig_P[3] + shift_3 * lattice_vec_a_orig_P[6]);
    point_set_orig[1] = point_set_orig[1] - (shift_1 * lattice_vec_a_orig_P[1] + shift_2 * lattice_vec_a_orig_P[4] + shift_3 * lattice_vec_a_orig_P[7]);
    point_set_orig[2] = point_set_orig[2] - (shift_1 * lattice_vec_a_orig_P[2] + shift_2 * lattice_vec_a_orig_P[5] + shift_3 * lattice_vec_a_orig_P[8]);

    in_k_material = FAME_Material_Handle(point_set_orig, lattice_vec_a_orig, Par);

    return in_k_material;
}

int FAME_Material_Locate_Index_Print(int* B_idx, PAR Par, char* type)
{
    int in_k_material;
    int i, j, k, m, idx, count = 0;
    int n = Par.mesh.grid_nums[0] * Par.mesh.grid_nums[1] * Par.mesh.grid_nums[2];
    int inout[Par.material.material_num], invP[3] = {0};
    realCPU Point_temp[3];
    realCPU lattice_vec_a_orig[9], lattice_vec_a_orig_P[9], invAP[9];

    /* lattice_vec_a_orig_P = Omega' * lattice_vec_a */
    for(int i = 0; i < 3; i++)
    {
        lattice_vec_a_orig_P[i]     = Par.lattice.Omega[count] * Par.lattice.lattice_vec_a[0] + Par.lattice.Omega[count + 1] * Par.lattice.lattice_vec_a[1] + Par.lattice.Omega[count + 2] * Par.lattice.lattice_vec_a[2];
        lattice_vec_a_orig_P[i + 3] = Par.lattice.Omega[count] * Par.lattice.lattice_vec_a[3] + Par.lattice.Omega[count + 1] * Par.lattice.lattice_vec_a[4] + Par.lattice.Omega[count + 2] * Par.lattice.lattice_vec_a[5];
        lattice_vec_a_orig_P[i + 6] = Par.lattice.Omega[count] * Par.lattice.lattice_vec_a[6] + Par.lattice.Omega[count + 1] * Par.lattice.lattice_vec_a[7] + Par.lattice.Omega[count + 2] * Par.lattice.lattice_vec_a[8];
        count = count + 3;
    }
    inv3(lattice_vec_a_orig_P, invAP);

    if (Par.lattice.Permutation[0] == 2 && Par.lattice.Permutation[1] == 3 && Par.lattice.Permutation[2] == 1)
    {
        invP[0] = 3;
        invP[1] = 1;
        invP[2] = 2;
    }
    else if (Par.lattice.Permutation[0] == 3 && Par.lattice.Permutation[1] == 1 && Par.lattice.Permutation[2] == 2)
    {
        invP[0] = 2;
        invP[1] = 3;
        invP[2] = 1;
    }
    else
    {
        invP[0] = Par.lattice.Permutation[0];
        invP[1] = Par.lattice.Permutation[1];
        invP[2] = Par.lattice.Permutation[2];
    }

    lattice_vec_a_orig[0] = lattice_vec_a_orig_P[(invP[0] - 1) * 3 + 0];
    lattice_vec_a_orig[1] = lattice_vec_a_orig_P[(invP[0] - 1) * 3 + 1];
    lattice_vec_a_orig[2] = lattice_vec_a_orig_P[(invP[0] - 1) * 3 + 2];
    lattice_vec_a_orig[3] = lattice_vec_a_orig_P[(invP[1] - 1) * 3 + 0];
    lattice_vec_a_orig[4] = lattice_vec_a_orig_P[(invP[1] - 1) * 3 + 1];
    lattice_vec_a_orig[5] = lattice_vec_a_orig_P[(invP[1] - 1) * 3 + 2];
    lattice_vec_a_orig[6] = lattice_vec_a_orig_P[(invP[2] - 1) * 3 + 0];
    lattice_vec_a_orig[7] = lattice_vec_a_orig_P[(invP[2] - 1) * 3 + 1];
    lattice_vec_a_orig[8] = lattice_vec_a_orig_P[(invP[2] - 1) * 3 + 2];


    for(k = 0; k < Par.mesh.grid_nums[2]; k++)
        for(j = 0; j < Par.mesh.grid_nums[1]; j++)
            for(i = 0; i < Par.mesh.grid_nums[0]; i++)
            {
                idx = k * Par.mesh.grid_nums[0] * Par.mesh.grid_nums[1] + j * Par.mesh.grid_nums[0] + i;

                FAME_Matrix_Grid(Point_temp, i, j, k, Par.mesh.mesh_len, type);

                in_k_material = FAME_Material_Locate(Point_temp, Par.lattice.Omega, lattice_vec_a_orig_P, invAP, lattice_vec_a_orig, Par);

                if(in_k_material != -1)
                    B_idx[idx + in_k_material * n] = 1;
            }

    return 0;
}

int FAME_Material_Locate_Index(MATERIAL* Material, PAR Par)
{
    int flag_fem = 0;
    int idx = Par.material.material_num * Par.mesh.grid_nums[0] * Par.mesh.grid_nums[1] * Par.mesh.grid_nums[2];

    if(strcmp(Par.material.material_type, "isotropic") == 0 || strcmp(Par.material.material_type, "anisotropic_diag") == 0)
    {
        Material->Binout = (int*) calloc (3 * idx, sizeof(int));
        FAME_Material_Locate_Index_Print(Material->Binout + idx * 0, Par, (char*)"Electric_x_point_set");
        FAME_Material_Locate_Index_Print(Material->Binout + idx * 1, Par, (char*)"Electric_y_point_set");
        FAME_Material_Locate_Index_Print(Material->Binout + idx * 2, Par, (char*)"Electric_z_point_set");
    }
    else if(strcmp(Par.material.material_type, "biisotropic") == 0)
    {
        Material->Binout = (int*) calloc (6 * idx, sizeof(int));
        FAME_Material_Locate_Index_Print(Material->Binout + idx * 0, Par, (char*)"Electric_x_point_set");
        FAME_Material_Locate_Index_Print(Material->Binout + idx * 1, Par, (char*)"Electric_y_point_set");
        FAME_Material_Locate_Index_Print(Material->Binout + idx * 2, Par, (char*)"Electric_z_point_set");
        FAME_Material_Locate_Index_Print(Material->Binout + idx * 3, Par, (char*)"Magnetic_x_point_set");
        FAME_Material_Locate_Index_Print(Material->Binout + idx * 4, Par, (char*)"Magnetic_y_point_set");
        FAME_Material_Locate_Index_Print(Material->Binout + idx * 5, Par, (char*)"Magnetic_z_point_set");
    }
    else if(strcmp(Par.material.material_type, "anisotropic") == 0 || strcmp(Par.material.material_type, "bianisotropic") == 0)
    {
        Material->BInOut = (int*) calloc (8 * idx, sizeof(int));
        FAME_Material_Locate_Index_Print(Material->BInOut + idx * 0, Par, (char*)"Electric_x_point_set");
        FAME_Material_Locate_Index_Print(Material->BInOut + idx * 1, Par, (char*)"Electric_y_point_set");
        FAME_Material_Locate_Index_Print(Material->BInOut + idx * 2, Par, (char*)"Electric_z_point_set");
        FAME_Material_Locate_Index_Print(Material->BInOut + idx * 6, Par, (char*)"xyz_point_set");
        FAME_Material_Locate_Index_Print(Material->BInOut + idx * 3, Par, (char*)"Magnetic_x_point_set");
        FAME_Material_Locate_Index_Print(Material->BInOut + idx * 4, Par, (char*)"Magnetic_y_point_set");
        FAME_Material_Locate_Index_Print(Material->BInOut + idx * 5, Par, (char*)"Magnetic_z_point_set");
        FAME_Material_Locate_Index_Print(Material->BInOut + idx * 7, Par, (char*)"none_xyz_point_set");

        int k;
        int N = Par.mesh.grid_nums[0] * Par.mesh.grid_nums[1] * Par.mesh.grid_nums[2];
        Material->BInOut_index_length = (int*) calloc(8, sizeof(int));
        Material->BInOut_index_length[0] = vec_sum(Material->BInOut, idx);
        Material->BInOut_index_length[1] = vec_sum(Material->BInOut + idx, idx) + Material->BInOut_index_length[0];
        Material->BInOut_index_length[2] = vec_sum(Material->BInOut + 2 * idx, idx) + Material->BInOut_index_length[1];
        Material->BInOut_index_length[3] = vec_sum(Material->BInOut + 3 * idx, idx) + Material->BInOut_index_length[2];
        Material->BInOut_index_length[4] = vec_sum(Material->BInOut + 4 * idx, idx) + Material->BInOut_index_length[3];
        Material->BInOut_index_length[5] = vec_sum(Material->BInOut + 5 * idx, idx) + Material->BInOut_index_length[4];
        Material->BInOut_index_length[6] = vec_sum(Material->BInOut + 6 * idx, idx) + Material->BInOut_index_length[5];
        Material->BInOut_index_length[7] = vec_sum(Material->BInOut + 7 * idx, idx) + Material->BInOut_index_length[6];
        Material->BInOut_index = (int*) calloc(Material->BInOut_index_length[7], sizeof(int));

        k = 0;
        for(int i = 0; i < 8; i++)
        {
            for(int j = 0; j < Par.material.material_num; j++)
            {
                for(int m = 0; m < N; m++)
                {
                    if(Material->BInOut[m + (j + i * Par.material.material_num) * N] == 1)
                    {
                        Material->BInOut_index[k] = m;
                        k++;
                    }
                }
            }
        }
    }
    else
    {
        Material->Binout = (int*) calloc (1 * idx, sizeof(int));
        FAME_Material_Locate_Index_Print(Material->Binout + idx * 0, Par, (char*)"Origin_point_set");
    }

    if(flag_fem)
        FAME_Material_Locate_Index_Print(Material->Binout + idx * 6, Par, (char*)"Origin_point_set");

    return 0;
}