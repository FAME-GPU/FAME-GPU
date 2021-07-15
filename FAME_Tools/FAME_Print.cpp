#include "FAME_Internal_Common.h"
// 2020-02-19

int FAME_Print(POPT Popt, char* data)
{
    int i;
    printf("# %s\n", data);

    if (strcmp(data, "Grid number") == 0)
    {
        for (i = 0; i < 3; i++)
            printf("% d ", Popt.mesh.grid_nums[i]);
        printf("\n");
    }

    else if (strcmp(data, "Part number") == 0)
    {
        printf("% d\n", Popt.recip_lattice.part_num);
    }

    else if (strcmp(data, "Material data name") == 0)
    {
        printf(" %s\n", Popt.material.data_name);
    }

    else if (strcmp(data, "Material type") == 0)
    {
        printf(" %s\n", Popt.material.material_type);
    }

    else if (strcmp(data, "Permittivity (inner material)") == 0)
    {
        for(i = 0; i < 9; i++)
        {
            printf("% lf ", Popt.material.ele_permitt_in[i]);
            if((i + 1) % 6 == 0)
                printf("\n");
        }
        printf("\n");
    }

    else if (strcmp(data, "Permittivity (outer material)") == 0)
    {
        printf("% lf\n", Popt.material.ele_permitt_out);
    }

    else if (strcmp(data, "Permeability (inner material)") == 0)
    {
        for(i = 0; i < 9; i++)
        {
            printf("% lf ", Popt.material.mag_permeab_in[i]);
            if((i + 1) % 6 == 0)
                printf("\n");
        }
        printf("\n");
    }

    else if (strcmp(data, "Permeability (outer material)") == 0)
    {
        printf("% lf\n", Popt.material.mag_permeab_out);
    }

    else if (strcmp(data, "Reciprocity (inner material) (must given when biisotropic)") == 0)
    {
        for(i = 0; i < 9; i++)
        {
            printf("% lf ", Popt.material.reciprocity_in[i]);
            if((i + 1) % 6 == 0)
                printf("\n");
        }
        printf("\n");
    }

    else if (strcmp(data, "Reciprocity (outer material) (must given when biisotropic)") == 0)
    {
        printf("% lf\n", Popt.material.reciprocity_out);
    }

    else if (strcmp(data, "Chirality (inner material) (must given when biisotropic)") == 0)
    {
        for(i = 0; i < 9; i++)
        {
            printf("% lf ", Popt.material.chirality_in[i]);
            if((i + 1) % 6 == 0)
                printf("\n");
        }
        printf("\n");
    }

    else if (strcmp(data, "Chirality (outer material) (must given when biisotropic)") == 0)
    {
        printf("% lf\n", Popt.material.chirality_out);
    }

    else if (strcmp(data, "Desired eigenpair number") == 0)
    {
        printf("% d\n", Popt.es.nwant);
    }

    else if (strcmp(data, "Dimension of Krylov subspace for Lanczos") == 0)
    {
        printf("% d\n", Popt.es.nstep);
    }

    else if (strcmp(data, "Tolerance of Lanczos") == 0)
    {
        printf("% e\n", Popt.es.tol);
    }

    else if (strcmp(data, "Maximum restart number of Lanczos") == 0)
    {
        printf("% d\n", Popt.es.maxit);
    }

    else if (strcmp(data, "Tolerance of linear solver") == 0)
    {
        printf("% e\n", Popt.ls.tol);
    }

    else if (strcmp(data, "Maximum iteration number of linear solver") == 0)
    {
        printf("% d\n", Popt.ls.maxit);
    }

    else if(strcmp(data, "Device") == 0)
    {
        printf("% d\n", Popt.flag.device);
    }

    else if(strcmp(data, "Printf User Option") == 0)
    {
        printf("% d\n", Popt.flag.printf_user_option);
    }

    else if(strcmp(data, "Printf Parameter") == 0)
    {
        printf("% d\n", Popt.flag.printf_parameter);
    }

    else if(strcmp(data, "Create Parameter") == 0)
    {
        printf("% d\n", Popt.flag.create_parameter);
    }

    else if(strcmp(data, "Create B_inout") == 0)
    {
        printf("% d\n", Popt.flag.create_B_inout);
    }

    else if(strcmp(data, "Create Wave Vector") == 0)
    {
        printf("% d\n", Popt.flag.create_wave_vector);
    }

    else if(strcmp(data, "Save Eigen Vector") == 0)
    {
        printf("% d\n", Popt.flag.save_eigen_vector);
    }

    else if(strcmp(data, "Grid nums Max") == 0)
    {
        printf("% d\n", Popt.flag.grid_nums_max);
    }

    else if(strcmp(data, "Sphere/Cylinder radius adjustment") == 0)
    {
        printf("% d % lf\n", Popt.flag.radius_adjustment, Popt.flag.radius);
    }
    
    else
        printf("Error\n");

    return 0;
}


int FAME_Print(PAR Par, char* data)
{
    int i, j;
    printf("# %s\n", data);

    if (strcmp(data, "Grid number") == 0)
    {
        for (i = 0; i < 3; i++)
            printf("% d ", Par.mesh.grid_nums[i]);
        printf("\n");
    }

    else if (strcmp(data, "Edge length") == 0)
    {
        for (i = 0; i < 3; i++)
            printf("% lf ", Par.mesh.edge_len[i]);
        printf("\n");
    }

    else if (strcmp(data, "Mesh length") == 0)
    {
        for (i = 0; i < 3; i++)
            printf("% lf ", Par.mesh.mesh_len[i]);
        printf("\n");
    }

    else if (strcmp(data, "Lattice type") == 0)
    {
        printf(" %s\n", Par.lattice.lattice_type);
    }

    else if (strcmp(data, "Lattice constant") == 0)
    {
        printf("% lf % lf % lf % lf % lf % lf\n", \
               Par.lattice.lattice_constant.a, \
               Par.lattice.lattice_constant.b, \
               Par.lattice.lattice_constant.c, \
               Par.lattice.lattice_constant.alpha, \
               Par.lattice.lattice_constant.beta, \
               Par.lattice.lattice_constant.gamma);
    }

    else if (strcmp(data, "Permutation") == 0)
    {
        for (i = 0; i < 3; i++)
            printf("% d ", Par.lattice.Permutation[i]);
        printf("\n");
    }

    else if (strcmp(data, "Length a") == 0)
    {
        printf("% lf % lf % lf\n", \
               Par.lattice.lattice_constant.length_a1, \
               Par.lattice.lattice_constant.length_a2, \
               Par.lattice.lattice_constant.length_a3);
    }

    else if (strcmp(data, "Lattice vector a (comp)") == 0)
    {
        printf("% lf \t % lf \t % lf\n", Par.lattice.lattice_vec_a[0], Par.lattice.lattice_vec_a[3], Par.lattice.lattice_vec_a[6]);
        printf("% lf \t % lf \t % lf\n", Par.lattice.lattice_vec_a[1], Par.lattice.lattice_vec_a[4], Par.lattice.lattice_vec_a[7]);
        printf("% lf \t % lf \t % lf\n", Par.lattice.lattice_vec_a[2], Par.lattice.lattice_vec_a[5], Par.lattice.lattice_vec_a[8]);
    }

    else if (strcmp(data, "Lattice vector a (orig)") == 0)
    {
        printf("% lf \t % lf \t % lf\n", Par.lattice.lattice_vec_a_orig[0], Par.lattice.lattice_vec_a_orig[3], Par.lattice.lattice_vec_a_orig[6]);
        printf("% lf \t % lf \t % lf\n", Par.lattice.lattice_vec_a_orig[1], Par.lattice.lattice_vec_a_orig[4], Par.lattice.lattice_vec_a_orig[7]);
        printf("% lf \t % lf \t % lf\n", Par.lattice.lattice_vec_a_orig[2], Par.lattice.lattice_vec_a_orig[5], Par.lattice.lattice_vec_a_orig[8]);
    }

    else if (strcmp(data, "Omega") == 0)
    {
        printf("% lf \t % lf \t % lf\n", Par.lattice.Omega[0], Par.lattice.Omega[3], Par.lattice.Omega[6]);
        printf("% lf \t % lf \t % lf\n", Par.lattice.Omega[1], Par.lattice.Omega[4], Par.lattice.Omega[7]);
        printf("% lf \t % lf \t % lf\n", Par.lattice.Omega[2], Par.lattice.Omega[5], Par.lattice.Omega[8]);
    }

    else if (strcmp(data, "Theta") == 0)
    {
        printf("% lf % lf % lf\n", \
               Par.lattice.lattice_constant.theta_1, \
               Par.lattice.lattice_constant.theta_2, \
               Par.lattice.lattice_constant.theta_3);
    }

    else if (strcmp(data, "Flag") == 0)
    {
        printf(" %s\n", Par.lattice.lattice_constant.flag);
    }

    else if (strcmp(data, "M") == 0)
    {
        printf("% d % d % d % d\n", \
               Par.lattice.lattice_constant.m1, \
               Par.lattice.lattice_constant.m2, \
               Par.lattice.lattice_constant.m3, \
               Par.lattice.lattice_constant.m4);
    }

    else if (strcmp(data, "t1 t2 t3 t4") == 0)
    {
        for( i = 0; i < 3; i++ )
            printf("% lf % lf % lf % lf\n", \
                    Par.lattice.lattice_constant.t1[i], \
                    Par.lattice.lattice_constant.t2[i], \
                    Par.lattice.lattice_constant.t3[i], \
                    Par.lattice.lattice_constant.t4[i]);

    }

    else if (strcmp(data, "Part number") == 0)
    {
        printf("% d\n", Par.recip_lattice.part_num);
    }

    else if (strcmp(data, "Brillouin zone Path string") == 0)
    {
        printf(" %s\n", Par.recip_lattice.path_string);
    }

    else if (strcmp(data, "Wave Vector Number") == 0)
    {
        printf("% d\n", Par.recip_lattice.Wave_vec_num);
    }

    else if (strcmp(data, "Material data name") == 0)
    {
        printf(" %s\n", Par.material.data_name);
    }

    else if (strcmp(data, "Material number") == 0)
    {
        printf("% d\n", Par.material.material_num);
    }

    else if (strcmp(data, "Material type") == 0)
    {
        printf(" %s\n", Par.material.material_type);
    }

    else if (strcmp(data, "Sphere number") == 0)
    {
        for (i = 0; i < Par.material.material_num; i++)
            printf("% d\n", Par.material.sphere_num[i]);
    }

    else if (strcmp(data, "Sphere centers") == 0)
    {
        int sum = 0;
        int count;

        for (i = 0; i < Par.material.material_num; i++)
            sum += Par.material.sphere_num[i];

        count = 0;
        j = 0;
        for (i = 1; i <= sum * 3; i++)
        {
            printf("% lf ", Par.material.sphere_centers[i - 1]);
            count++;
            if (count == Par.material.sphere_num[j] * 3)
            {
                count = 0;
                j++;
                printf("\n");
            }
            if(count % 6 == 0)
                printf("\n");
        }
    }

    else if (strcmp(data, "Sphere radius") == 0)
    {
        int sum = 0;
        int count;

        for (i = 0; i < Par.material.material_num; i++)
            sum += Par.material.sphere_num[i];

        count = 0;
        j = 0;
        for (i = 1; i <= sum; i++)
        {
            printf("% lf ", Par.material.sphere_radius[i - 1]);
            count++;
            if (count == Par.material.sphere_num[j])
            {
                count = 0;
                j++;
                printf("\n");
            }
            if(count % 6 == 0)
                printf("\n");
        }
    }

    else if (strcmp(data, "Cylinder number") == 0)
    {
        for (i = 0; i < Par.material.material_num; i++)
            printf("% d\n", Par.material.cylinder_num[i]);
    }

    else if (strcmp(data, "Cylinder top centers") == 0)
    {
        int sum = 0;
        int count;

        for (i = 0; i < Par.material.material_num; i++)
            sum += Par.material.cylinder_num[i];

        count = 0;
        j = 0;
        for (i = 1; i <= sum * 3; i++)
        {
            printf("% lf ", Par.material.cylinder_top_centers[i - 1]);
            count++;
            if (count == Par.material.cylinder_num[j] * 3)
            {
                count = 0;
                j++;
                printf("\n");
            }
            if(count % 6 == 0)
                printf("\n");
        }
    }

    else if (strcmp(data, "Cylinder bot centers") == 0)
    {
        int sum = 0;
        int count;

        for (i = 0; i < Par.material.material_num; i++)
            sum += Par.material.cylinder_num[i];

        count = 0;
        j = 0;
        for (i = 1; i <= sum * 3; i++)
        {
            printf("% lf ", Par.material.cylinder_bot_centers[i - 1]);
            count++;
            if (count == Par.material.cylinder_num[j] * 3)
            {
                count = 0;
                j++;
                printf("\n");
            }
            if(count % 6 == 0)
                printf("\n");
        }
    }

    else if (strcmp(data, "Cylinder radius") == 0)
    {
        int sum = 0;
        int count;

        for (i = 0; i < Par.material.material_num; i++)
            sum += Par.material.cylinder_num[i];

        count = 0;
        j = 0;
        for (i = 1; i <= sum; i++)
        {
            printf("% lf ", Par.material.cylinder_radius[i - 1]);
            count++;
            if (count == Par.material.cylinder_num[j])
            {
                count = 0;
                j++;
                printf("\n");
            }
            if(count % 6 == 0)
                printf("\n");
        }
    }

    else if (strcmp(data, "Permittivity (inner material)") == 0)
    {
        for(i = 0; i < 9; i++)
        {
            printf("% lf ", Par.material.ele_permitt_in[i]);
            if((i + 1) % 6 == 0)
                printf("\n");
        }
        printf("\n");
    }

    else if (strcmp(data, "Permittivity (outer material)") == 0)
    {
        printf("% lf\n", Par.material.ele_permitt_out);
    }


    else if (strcmp(data, "Permeability (inner material)") == 0)
    {
        for(i = 0; i < 9; i++)
        {
            printf("% lf ", Par.material.mag_permeab_in[i]);
            if((i + 1) % 6 == 0)
                printf("\n");
        }
        printf("\n");
    }

    else if (strcmp(data, "Permeability (outer material)") == 0)
    {
        printf("% lf\n", Par.material.mag_permeab_out);
    }

    else if (strcmp(data, "Reciprocity (inner material) (must given when biisotropic)") == 0)
    {
        for(i = 0; i < 9; i++)
        {
            printf("% lf ", Par.material.reciprocity_in[i]);
            if((i + 1) % 6 == 0)
                printf("\n");
        }
        printf("\n");
    }

    else if (strcmp(data, "Reciprocity (outer material) (must given when biisotropic)") == 0)
    {
        printf("% lf\n", Par.material.reciprocity_out);
    }

    else if (strcmp(data, "Chirality (inner material) (must given when biisotropic)") == 0)
    {
        for(i = 0; i < 9; i++)
        {
            printf("% lf ", Par.material.chirality_in[i]);
            if((i + 1) % 6 == 0)
                printf("\n");
        }
        printf("\n");
    }

    else if (strcmp(data, "Chirality (outer material) (must given when biisotropic)") == 0)
    {
        printf("% lf\n", Par.material.chirality_out);
    }

    else if (strcmp(data, "Desired eigenpair number") == 0)
    {
        printf("% d\n", Par.es.nwant);
    }

    else if (strcmp(data, "Dimension of Krylov subspace for Lanczos") == 0)
    {
        printf("% d\n", Par.es.nstep);
    }

    else if (strcmp(data, "Tolerance of Lanczos") == 0)
    {
        printf("% e\n", Par.es.tol);
    }

    else if (strcmp(data, "Maximum restart number of Lanczos") == 0)
    {
        printf("% d\n", Par.es.maxit);
    }

    else if (strcmp(data, "Tolerance of linear solver") == 0)
    {
        printf("% e\n", Par.ls.tol);
    }

    else if (strcmp(data, "Maximum iteration number of linear solver") == 0)
    {
        printf("% d\n", Par.ls.maxit);
    }

    else if(strcmp(data, "Device") == 0)
    {
        printf("% d\n", Par.flag.device);
    }

    else if(strcmp(data, "Printf User Option") == 0)
    {
        printf("% d\n", Par.flag.printf_user_option);
    }

    else if(strcmp(data, "Printf Parameter") == 0)
    {
        printf("% d\n", Par.flag.printf_parameter);
    }

    else if(strcmp(data, "Create Parameter") == 0)
    {
        printf("% d\n", Par.flag.create_parameter);
    }

    else if(strcmp(data, "Create B_inout") == 0)
    {
        printf("% d\n", Par.flag.create_B_inout);
    }

    else if(strcmp(data, "Create Wave Vector") == 0)
    {
        printf("% d\n", Par.flag.create_wave_vector);
    }

    else if(strcmp(data, "Save Eigen Vector") == 0)
    {
        printf("% d\n", Par.flag.save_eigen_vector);
    }

    else if(strcmp(data, "Grid nums Max") == 0)
    {
        printf("% d\n", Par.flag.grid_nums_max);
    }

    else if(strcmp(data, "Sphere/Cylinder radius adjustment") == 0)
    {
        printf("% d % lf\n", Par.flag.radius_adjustment, Par.flag.radius);
    }

    else
        printf("Error\n");

    return 0;
}


int FAME_Print(PAR Par, char* data, FILE* fp)
{
    int i, j;
    fprintf(fp, "# %s\n", data);

    if (strcmp(data, "Grid number") == 0)
    {
        for (i = 0; i < 3; i++)
            fprintf(fp, "% d ", Par.mesh.grid_nums[i]);
        fprintf(fp, "\n");
    }

    else if (strcmp(data, "Edge length") == 0)
    {
        for (i = 0; i < 3; i++)
            fprintf(fp, "% lf ", Par.mesh.edge_len[i]);
        fprintf(fp, "\n");
    }

    else if (strcmp(data, "Mesh length") == 0)
    {
        for (i = 0; i < 3; i++)
            fprintf(fp, "% lf ", Par.mesh.mesh_len[i]);
        fprintf(fp, "\n");
    }

    else if (strcmp(data, "Lattice type") == 0)
    {
        fprintf(fp, " %s\n", Par.lattice.lattice_type);
    }

    else if (strcmp(data, "Lattice constant") == 0)
    {
        fprintf(fp, "% lf % lf % lf % lf % lf % lf\n", \
               Par.lattice.lattice_constant.a, \
               Par.lattice.lattice_constant.b, \
               Par.lattice.lattice_constant.c, \
               Par.lattice.lattice_constant.alpha, \
               Par.lattice.lattice_constant.beta, \
               Par.lattice.lattice_constant.gamma);
    }

    else if (strcmp(data, "Permutation") == 0)
    {
        for (i = 0; i < 3; i++)
            fprintf(fp, "% d ", Par.lattice.Permutation[i]);
        fprintf(fp, "\n");
    }

    else if (strcmp(data, "Length a") == 0)
    {
        fprintf(fp, "% lf % lf % lf\n", \
               Par.lattice.lattice_constant.length_a1, \
               Par.lattice.lattice_constant.length_a2, \
               Par.lattice.lattice_constant.length_a3);
    }
    
    else if (strcmp(data, "Lattice vector a (comp)") == 0)
    {
        fprintf(fp, "% lf \t % lf \t % lf\n", Par.lattice.lattice_vec_a[0], Par.lattice.lattice_vec_a[3], Par.lattice.lattice_vec_a[6]);
        fprintf(fp, "% lf \t % lf \t % lf\n", Par.lattice.lattice_vec_a[1], Par.lattice.lattice_vec_a[4], Par.lattice.lattice_vec_a[7]);
        fprintf(fp, "% lf \t % lf \t % lf\n", Par.lattice.lattice_vec_a[2], Par.lattice.lattice_vec_a[5], Par.lattice.lattice_vec_a[8]);
    }

    else if (strcmp(data, "Lattice vector a (orig)") == 0)
    {
        fprintf(fp, "% lf \t % lf \t % lf\n", Par.lattice.lattice_vec_a_orig[0], Par.lattice.lattice_vec_a_orig[3], Par.lattice.lattice_vec_a_orig[6]);
        fprintf(fp, "% lf \t % lf \t % lf\n", Par.lattice.lattice_vec_a_orig[1], Par.lattice.lattice_vec_a_orig[4], Par.lattice.lattice_vec_a_orig[7]);
        fprintf(fp, "% lf \t % lf \t % lf\n", Par.lattice.lattice_vec_a_orig[2], Par.lattice.lattice_vec_a_orig[5], Par.lattice.lattice_vec_a_orig[8]);
    }

    else if (strcmp(data, "Omega") == 0)
    {
        fprintf(fp, "% lf \t % lf \t % lf\n", Par.lattice.Omega[0], Par.lattice.Omega[3], Par.lattice.Omega[6]);
        fprintf(fp, "% lf \t % lf \t % lf\n", Par.lattice.Omega[1], Par.lattice.Omega[4], Par.lattice.Omega[7]);
        fprintf(fp, "% lf \t % lf \t % lf\n", Par.lattice.Omega[2], Par.lattice.Omega[5], Par.lattice.Omega[8]);
    }

    else if (strcmp(data, "Theta") == 0)
    {
        fprintf(fp, "% lf % lf % lf\n", \
               Par.lattice.lattice_constant.theta_1, \
               Par.lattice.lattice_constant.theta_2, \
               Par.lattice.lattice_constant.theta_3);
    }

    else if (strcmp(data, "Flag") == 0)
    {
        fprintf(fp, " %s\n", Par.lattice.lattice_constant.flag);
    }

    else if (strcmp(data, "M") == 0)
    {
        fprintf(fp, "% d % d % d % d\n", \
               Par.lattice.lattice_constant.m1, \
               Par.lattice.lattice_constant.m2, \
               Par.lattice.lattice_constant.m3, \
               Par.lattice.lattice_constant.m4);
    }

    else if (strcmp(data, "t1 t2 t3 t4") == 0)
    {
        for( i = 0; i < 3; i++ )
            fprintf(fp, "% lf % lf % lf % lf\n", \
                    Par.lattice.lattice_constant.t1[i], \
                    Par.lattice.lattice_constant.t2[i], \
                    Par.lattice.lattice_constant.t3[i], \
                    Par.lattice.lattice_constant.t4[i]);

    }

    else if (strcmp(data, "Part number") == 0)
    {
        fprintf(fp, "% d\n", Par.recip_lattice.part_num);
    }

    else if (strcmp(data, "Brillouin zone Path string") == 0)
    {
        fprintf(fp, " %s\n", Par.recip_lattice.path_string);
    }

    else if (strcmp(data, "Wave Vector Number") == 0)
    {
        fprintf(fp, "% d\n", Par.recip_lattice.Wave_vec_num);
    }

    else if (strcmp(data, "Material data name") == 0)
    {
        fprintf(fp, " %s\n", Par.material.data_name);
    }

    else if (strcmp(data, "Material number") == 0)
    {
        fprintf(fp, "% d\n", Par.material.material_num);
    }

    else if (strcmp(data, "Material type") == 0)
    {
        fprintf(fp, " %s\n", Par.material.material_type);
    }

    else if (strcmp(data, "Sphere number") == 0)
    {
        for (i = 0; i < Par.material.material_num; i++)
            fprintf(fp, "% d\n", Par.material.sphere_num[i]);
    }

    else if (strcmp(data, "Sphere centers") == 0)
    {
        int sum = 0;
        int count;

        for (i = 0; i < Par.material.material_num; i++)
            sum += Par.material.sphere_num[i];

        count = 0;
        j = 0;
        for (i = 1; i <= sum * 3; i++)
        {
            fprintf(fp, "% lf ", Par.material.sphere_centers[i - 1]);
            count++;
            if (count == Par.material.sphere_num[j] * 3)
            {
                count = 0;
                j++;
                fprintf(fp, "\n");
            }
            if(count % 6 == 0)
                fprintf(fp, "\n");
        }
    }

    else if (strcmp(data, "Sphere radius") == 0)
    {
        int sum = 0;
        int count;

        for (i = 0; i < Par.material.material_num; i++)
            sum += Par.material.sphere_num[i];

        count = 0;
        j = 0;
        for (i = 1; i <= sum; i++)
        {
            fprintf(fp, "% lf ", Par.material.sphere_radius[i - 1]);
            count++;
            if (count == Par.material.sphere_num[j])
            {
                count = 0;
                j++;
                fprintf(fp, "\n");
            }
            if(count % 6 == 0)
                fprintf(fp, "\n");
        }
    }

    else if (strcmp(data, "Cylinder number") == 0)
    {
        for (i = 0; i < Par.material.material_num; i++)
            fprintf(fp, "% d\n", Par.material.cylinder_num[i]);
    }

    else if (strcmp(data, "Cylinder top centers") == 0)
    {
        int sum = 0;
        int count;

        for (i = 0; i < Par.material.material_num; i++)
            sum += Par.material.cylinder_num[i];

        count = 0;
        j = 0;
        for (i = 1; i <= sum * 3; i++)
        {
            fprintf(fp, "% lf ", Par.material.cylinder_top_centers[i - 1]);
            count++;
            if (count == Par.material.cylinder_num[j] * 3)
            {
                count = 0;
                j++;
                fprintf(fp, "\n");
            }
            if(count % 6 == 0)
                fprintf(fp, "\n");
        }
    }

    else if (strcmp(data, "Cylinder bot centers") == 0)
    {
        int sum = 0;
        int count;

        for (i = 0; i < Par.material.material_num; i++)
            sum += Par.material.cylinder_num[i];

        count = 0;
        j = 0;
        for (i = 1; i <= sum * 3; i++)
        {
            fprintf(fp, "% lf ", Par.material.cylinder_bot_centers[i - 1]);
            count++;
            if (count == Par.material.cylinder_num[j] * 3)
            {
                count = 0;
                j++;
                fprintf(fp, "\n");
            }
            if(count % 6 == 0)
                fprintf(fp, "\n");
        }
    }

    else if (strcmp(data, "Cylinder radius") == 0)
    {
        int sum = 0;
        int count;

        for (i = 0; i < Par.material.material_num; i++)
            sum += Par.material.cylinder_num[i];

        count = 0;
        j = 0;
        for (i = 1; i <= sum; i++)
        {
            fprintf(fp, "% lf ", Par.material.cylinder_radius[i - 1]);
            count++;
            if (count == Par.material.cylinder_num[j])
            {
                count = 0;
                j++;
                fprintf(fp, "\n");
            }
            if(count % 6 == 0)
                fprintf(fp, "\n");
        }
    }

    else if (strcmp(data, "Permittivity (inner material)") == 0)
    {
        for(i = 0; i < 9; i++)
        {
            fprintf(fp, "% lf ", Par.material.ele_permitt_in[i]);
            if((i + 1) % 6 == 0)
                fprintf(fp, "\n");
        }
        fprintf(fp, "\n");
    }

    else if (strcmp(data, "Permittivity (outer material)") == 0)
    {
        fprintf(fp, "% lf\n", Par.material.ele_permitt_out);
    }

    else if (strcmp(data, "Desired eigenpair number") == 0)
    {
        fprintf(fp, "% d\n", Par.es.nwant);
    }

    else if (strcmp(data, "Dimension of Krylov subspace for Lanczos") == 0)
    {
        fprintf(fp, "% d\n", Par.es.nstep);
    }

    else if (strcmp(data, "Tolerance of Lanczos") == 0)
    {
        fprintf(fp, "% e\n", Par.es.tol);
    }

    else if (strcmp(data, "Maximum restart number of Lanczos") == 0)
    {
        fprintf(fp, "% d\n", Par.es.maxit);
    }

    else if (strcmp(data, "Tolerance of linear solver") == 0)
    {
        fprintf(fp, "% e\n", Par.ls.tol);
    }

    else if (strcmp(data, "Maximum iteration number of linear solver") == 0)
    {
        fprintf(fp, "% d\n", Par.ls.maxit);
    }

    else if(strcmp(data, "Device") == 0)
    {
        fprintf(fp, "% d\n", Par.flag.device);
    }

    else if(strcmp(data, "Printf User Option") == 0)
    {
        fprintf(fp, "% d\n", Par.flag.printf_user_option);
    }

    else if(strcmp(data, "Printf Parameter") == 0)
    {
        fprintf(fp, "% d\n", Par.flag.printf_parameter);
    }

    else if(strcmp(data, "Create Parameter") == 0)
    {
        fprintf(fp, "% d\n", Par.flag.create_parameter);
    }

    else if(strcmp(data, "Create B_inout") == 0)
    {
        fprintf(fp, "% d\n", Par.flag.create_B_inout);
    }

    else if(strcmp(data, "Create Wave Vector") == 0)
    {
        fprintf(fp, "% d\n", Par.flag.create_wave_vector);
    }

    else if(strcmp(data, "Save Eigen Vector") == 0)
    {
        fprintf(fp, "% d\n", Par.flag.save_eigen_vector);
    }

    else if(strcmp(data, "Grid nums Max") == 0)
    {
        fprintf(fp, "% d\n", Par.flag.grid_nums_max);
    }

    else if(strcmp(data, "Sphere/Cylinder radius adjustment") == 0)
    {
        fprintf(fp, "% d % lf\n", Par.flag.radius_adjustment, Par.flag.radius);
    }

    else
        fprintf(fp, "Error\n");

    return 0;
}