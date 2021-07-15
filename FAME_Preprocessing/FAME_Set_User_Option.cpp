#include "FAME_Internal_Common.h"
// 2020-02-19

int FAME_Set_User_Option( POPT *Popt )
{
    int i;
    char str[100];

    FILE *fp;
    fp = fopen("Popt.txt", "r");
    assert(fp != NULL);

    while(fgets(str, 100, fp) != NULL)
    {
        str[strlen(str) - 1] = '\0';
       
        // ======================== Mesh information ========================
        if(strcmp(str, "# Grid number") == 0)
        {
            for(i = 0; i < 3; i++)
                fscanf(fp, "%d", &Popt->mesh.grid_nums[i]);
        }

        // ================= Reciprocal lattice information =================
        else if(strcmp(str, "# Part number") == 0)
        {
            fscanf(fp, "%d", &Popt->recip_lattice.part_num);
        }

        // ====================== Material information ======================
        else if(strcmp(str, "# Material data name") == 0)
        {
            fscanf(fp, "%s", Popt->material.data_name);
        }

        else if(strcmp(str, "# Material type") == 0)
        {
            fscanf(fp, "%s", Popt->material.material_type);
        }

        else if(strcmp(str, "# Permittivity (inner material)") == 0)
        {
            Popt->material.ele_permitt_in = (realCPU*) calloc(18, sizeof(realCPU));

            for(int i = 0; i < 9; i++)
                fscanf(fp, "%lf", &Popt->material.ele_permitt_in[i]);

            for(int i = 0; i < 9; i++)
                fscanf(fp, "%lf", &Popt->material.ele_permitt_in[i + 9]);
            
        }

        else if(strcmp(str, "# Permittivity (outer material)") == 0)
        {
            fscanf(fp, "%lf", &Popt->material.ele_permitt_out);                 
        }

        else if(strcmp(str, "# Permeability (inner material)") == 0)
        {

            Popt->material.mag_permeab_in = (realCPU*) calloc(18, sizeof(realCPU));

            for(int i = 0; i < 9; i++)
                fscanf(fp, "%lf", &Popt->material.mag_permeab_in[i]);

            for(int i = 0; i < 9; i++)
                fscanf(fp, "%lf", &Popt->material.mag_permeab_in[i + 9]);
        }

        else if(strcmp(str, "# Permeability (outer material)") == 0)
        {
            fscanf(fp, "%lf", &Popt->material.mag_permeab_out);
        }

        else if(strcmp(str, "# Reciprocity (inner material) (must given when bi-isotropic and bi-anisotropic)") == 0)
        {

            Popt->material.reciprocity_in = (realCPU*) calloc(9, sizeof(realCPU));
            
            for(int i = 0; i < 9; i++)
                fscanf(fp, "%lf", &Popt->material.reciprocity_in[i]);
              
        }

        else if(strcmp(str, "# Reciprocity (outer material) (must given when bi-isotropic and bi-anisotropic)") == 0)
        {
            fscanf(fp, "%lf", &Popt->material.reciprocity_out);
        }

        else if(strcmp(str, "# Chirality (inner material) (must given when bi-isotropic and bi-anisotropic)") == 0)
        {

            Popt->material.chirality_in = (realCPU*) calloc(18, sizeof(realCPU));

            for(int i = 0; i < 9; i++)
                fscanf(fp, "%lf", &Popt->material.chirality_in[i]);

            for(int i = 0; i < 9; i++)
                fscanf(fp, "%lf", &Popt->material.chirality_in[i + 9]);
        }

        else if(strcmp(str, "# Chirality (outer material) (must given when bi-isotropic and bi-anisotropic)") == 0)
        {
            fscanf(fp, "%lf", &Popt->material.chirality_out);
        }

        // ======================= Solver information =======================
        else if(strcmp(str, "# Desired eigenpair number") == 0)
        {
            fscanf(fp, "%d", &Popt->es.nwant);
        }

        else if(strcmp(str, "# Dimension of Krylov subspace for Lanczos") == 0)
        {
            fscanf(fp, "%d", &Popt->es.nstep);
        }

        else if(strcmp(str, "# Maximum restart number of Lanczos") == 0)
        {
            fscanf(fp, "%d", &Popt->es.maxit);
        }

        else if(strcmp(str, "# Maximum iteration number of linear solver") == 0)
        {
            fscanf(fp, "%d", &Popt->ls.maxit);
        }

        // ========================== Flag setting ==========================
        else if(strcmp(str, "# Device") == 0)
        {
            fscanf(fp, "%d", &Popt->flag.device);
        }

        else if(strcmp(str, "# Printf User Option") == 0)
        {
            fscanf(fp, "%d", &Popt->flag.printf_user_option);
        }

        else if(strcmp(str, "# Printf Parameter") == 0)
        {
            fscanf(fp, "%d", &Popt->flag.printf_parameter);
        }

        else if(strcmp(str, "# Create Parameter") == 0)
        {
            fscanf(fp, "%d", &Popt->flag.create_parameter);
        }

        else if(strcmp(str, "# Create B_inout") == 0)
        {
            fscanf(fp, "%d", &Popt->flag.create_B_inout);
        }

        else if(strcmp(str, "# Create Wave Vector") == 0)
        {
            fscanf(fp, "%d", &Popt->flag.create_wave_vector);
        }

        else if(strcmp(str, "# Save Eigen Vector") == 0)
        {
            fscanf(fp, "%d", &Popt->flag.save_eigen_vector);
        }

        else if(strcmp(str, "# Grid nums Max") == 0)
        {
            fscanf(fp, "%d", &Popt->flag.grid_nums_max);
            fscanf(fp, "%d", &Popt->flag.mem_size);
        }

        else if(strcmp(str, "# Sphere/Cylinder radius adjustment") == 0)
        {
            fscanf(fp, "%d %lf", &Popt->flag.radius_adjustment, &Popt->flag.radius);
        }

        else
        {

        }

    }
    fclose(fp);

    return 0;
}


int FAME_Set_User_Option_Single( POPT *Popt )
{
    int i;
    char str[100];

    FILE *fp;
    fp = fopen("Popt.txt", "r");
    assert(fp != NULL);

    while(fgets(str, 100, fp) != NULL)
    {
        str[strlen(str) - 1] = '\0';
       
        // ======================== Mesh information ========================
        if(strcmp(str, "# Grid number") == 0)
        {
            for(i = 0; i < 3; i++)
                fscanf(fp, "%d", &Popt->mesh.grid_nums[i]);
        }

        // ================= Reciprocal lattice information =================
        else if(strcmp(str, "# Part number") == 0)
        {
            fscanf(fp, "%d", &Popt->recip_lattice.part_num);
        }

        // ====================== Material information ======================
        else if(strcmp(str, "# Material data name") == 0)
        {
            fscanf(fp, "%s", Popt->material.data_name);
        }

        else if(strcmp(str, "# Material type") == 0)
        {
            fscanf(fp, "%s", Popt->material.material_type);
        }

        else if(strcmp(str, "# Permittivity (inner material)") == 0)
        {
            Popt->material.ele_permitt_in = (realCPU*) calloc(18, sizeof(realCPU));

            for(int i = 0; i < 9; i++)
                fscanf(fp, "%f", &Popt->material.ele_permitt_in[i]);

            for(int i = 0; i < 9; i++)
                fscanf(fp, "%f", &Popt->material.ele_permitt_in[i + 9]);
            
        }

        else if(strcmp(str, "# Permittivity (outer material)") == 0)
        {
            fscanf(fp, "%f", &Popt->material.ele_permitt_out);                 
        }

        else if(strcmp(str, "# Permeability (inner material)") == 0)
        {

            Popt->material.mag_permeab_in = (realCPU*) calloc(18, sizeof(realCPU));

            for(int i = 0; i < 9; i++)
                fscanf(fp, "%f", &Popt->material.mag_permeab_in[i]);

            for(int i = 0; i < 9; i++)
                fscanf(fp, "%f", &Popt->material.mag_permeab_in[i + 9]);
        }

        else if(strcmp(str, "# Permeability (outer material)") == 0)
        {
            fscanf(fp, "%f", &Popt->material.mag_permeab_out);
        }

        else if(strcmp(str, "# Reciprocity (inner material) (must given when biisotropic and bi-anisotropic)") == 0)
        {

            Popt->material.reciprocity_in = (realCPU*) calloc(9, sizeof(realCPU));
            
            for(int i = 0; i < 9; i++)
                fscanf(fp, "%f", &Popt->material.reciprocity_in[i]);
                
        }

        else if(strcmp(str, "# Reciprocity (outer material) (must given when biisotropic and bi-anisotropic)") == 0)
        {
            fscanf(fp, "%f", &Popt->material.reciprocity_out);
        }

        else if(strcmp(str, "# Chirality (inner material) (must given when biisotropic and bi-anisotropic)") == 0)
        {

            Popt->material.chirality_in = (realCPU*) calloc(18, sizeof(realCPU));

            for(int i = 0; i < 9; i++)
                fscanf(fp, "%f", &Popt->material.chirality_in[i]);

            for(int i = 0; i < 9; i++)
                fscanf(fp, "%f", &Popt->material.chirality_in[i + 9]);
        }

        else if(strcmp(str, "# Chirality (outer material) (must given when biisotropic and bi-anisotropic)") == 0)
        {
            fscanf(fp, "%f", &Popt->material.chirality_out);
        }

        // ======================= Solver information =======================
        else if(strcmp(str, "# Desired eigenpair number") == 0)
        {
            fscanf(fp, "%d", &Popt->es.nwant);
        }

        else if(strcmp(str, "# Dimension of Krylov subspace for Lanczos") == 0)
        {
            fscanf(fp, "%d", &Popt->es.nstep);
        }

        else if(strcmp(str, "# Maximum restart number of Lanczos") == 0)
        {
            fscanf(fp, "%d", &Popt->es.maxit);
        }

        else if(strcmp(str, "# Maximum iteration number of linear solver") == 0)
        {
            fscanf(fp, "%d", &Popt->ls.maxit);
        }

        // ========================== Flag setting ==========================
        else if(strcmp(str, "# Device") == 0)
        {
            fscanf(fp, "%d", &Popt->flag.device);
        }

        else if(strcmp(str, "# Printf User Option") == 0)
        {
            fscanf(fp, "%d", &Popt->flag.printf_user_option);
        }

        else if(strcmp(str, "# Printf Parameter") == 0)
        {
            fscanf(fp, "%d", &Popt->flag.printf_parameter);
        }

        else if(strcmp(str, "# Create Parameter") == 0)
        {
            fscanf(fp, "%d", &Popt->flag.create_parameter);
        }

        else if(strcmp(str, "# Create B_inout") == 0)
        {
            fscanf(fp, "%d", &Popt->flag.create_B_inout);
        }

        else if(strcmp(str, "# Create Wave Vector") == 0)
        {
            fscanf(fp, "%d", &Popt->flag.create_wave_vector);
        }

        else if(strcmp(str, "# Save Eigen Vector") == 0)
        {
            fscanf(fp, "%d", &Popt->flag.save_eigen_vector);
        }

        else if(strcmp(str, "# Grid nums Max") == 0)
        {
            fscanf(fp, "%d", &Popt->flag.grid_nums_max);
            fscanf(fp, "%d", &Popt->flag.mem_size);
        }

        else if(strcmp(str, "# Sphere/Cylinder radius adjustment") == 0)
        {
            fscanf(fp, "%d %f", &Popt->flag.radius_adjustment, &Popt->flag.radius);
        }

        else
        {

        }

    }
    fclose(fp);

    return 0;
}
