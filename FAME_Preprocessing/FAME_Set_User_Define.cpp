#include "FAME_Internal_Common.h"
// 2020-02-19

int FAME_Set_User_Define( MATERIAL* Material, LATTICE* Lattice )
{
    int i;
    char str[100];
    int total_sphere = 0, total_cylinder = 0;
    char filepath[100] = "./FAME_Material_Data/";
    strcat(filepath, Material->data_name);
    strcat(filepath, ".txt");

    FILE *fp;
    fp = fopen(filepath, "r");

    if(fp == NULL)
    {
        printf("\033[40;31mFAME_Set_User_Define(19):\033[0m\n");
        printf("\033[40;31mThe material data '%s' is not found in %s.\033[0m\n", Material->data_name, filepath);
        printf("\033[40;31mPlease check material data name is correct in Popt.txt or the material data is in FAME_Material_Data.\033[0m\n");
        assert(0);
    }

    while(fgets(str, 100, fp) != NULL)
    {
        str[strlen(str) - 1] = '\0';
        if(strcmp(str, "# Data name") == 0)
        {
            fscanf(fp, "%s", Material->data_name);
        }

       else if(strcmp(str, "# Material number") == 0)
        {
            fscanf(fp, "%d", &Material->material_num);
            Material->sphere_num   = (int*) malloc(Material->material_num * sizeof(int));
            Material->cylinder_num = (int*) malloc(Material->material_num * sizeof(int));
        }

        else if(strcmp(str, "# Lattice type") == 0)
        {
            fscanf(fp, "%s", Lattice->lattice_type);
        }

        else if(strcmp(str, "# Lattice constant") == 0)
        {
            fscanf(fp, "%lf %lf %lf %lf %lf %lf", \
                   &Lattice->lattice_constant.a, &Lattice->lattice_constant.b, &Lattice->lattice_constant.c,\
                   &Lattice->lattice_constant.alpha, &Lattice->lattice_constant.beta, &Lattice->lattice_constant.gamma);
        }

        else if(strcmp(str, "# Sphere number") == 0)
        {
            for(i = 0; i < Material->material_num; i++)
            {
                fscanf(fp, "%d", &Material->sphere_num[i]);
                total_sphere += Material->sphere_num[i];
            }
            Material->sphere_centers = (realCPU*) malloc(total_sphere * 3 * sizeof(realCPU));
            Material->sphere_radius = (realCPU*) malloc(total_sphere * sizeof(realCPU));
        }

        else if(strcmp(str, "# Sphere centers") == 0)
        {
            for(i = 0; i < total_sphere * 3; i++)
                fscanf(fp, "%lf", &Material->sphere_centers[i]);
        }

        else if(strcmp(str, "# Sphere radius") == 0)
        {
            if(Material->flag_radius_adjustment)
            {
                for(i = 0; i < total_sphere; i++)
                    Material->sphere_radius[i] = Material->flag_radius;
            }
            else
            {
                for(i = 0; i < total_sphere; i++)
                    fscanf(fp, "%lf", &Material->sphere_radius[i]);
            }
        }

        else if(strcmp(str, "# Cylinder number") == 0)
        {
            total_cylinder = 0;
            for(i = 0; i < Material->material_num; i++)
            {
                fscanf(fp, "%d", &Material->cylinder_num[i]);
                total_cylinder += Material->cylinder_num[i];
            }
            Material->cylinder_top_centers = (realCPU*) malloc(total_cylinder * 3 * sizeof(realCPU));
            Material->cylinder_bot_centers = (realCPU*) malloc(total_cylinder * 3 * sizeof(realCPU));
            Material->cylinder_radius = (realCPU*) malloc(total_cylinder * sizeof(realCPU));
        }

        else if(strcmp(str, "# Cylinder top centers") == 0)
        {
            for(i = 0; i < total_cylinder * 3; i++)
                fscanf(fp, "%lf", &Material->cylinder_top_centers[i]);
        }

        else if(strcmp(str, "# Cylinder bot centers") == 0)
        {
            for(i = 0; i < total_cylinder * 3; i++)
                fscanf(fp, "%lf", &Material->cylinder_bot_centers[i]);
        }

        else if(strcmp(str, "# Cylinder radius") == 0)
        {
            if(Material->flag_radius_adjustment)
            {
                for(i = 0; i < total_cylinder; i++)
                    Material->cylinder_radius[i] = Material->flag_radius;
            }
            else
            {
                for(i = 0; i < total_cylinder; i++)
                    fscanf(fp, "%lf", &Material->cylinder_radius[i]);
            }
        }
    }
    fclose(fp);
    return 0;
}



int FAME_Set_User_Define_Single( MATERIAL* Material, LATTICE* Lattice )
{
    int i;
    char str[100];
    int total_sphere = 0, total_cylinder = 0;
    char filepath[100] = "./FAME_Material_Data/";
    strcat(filepath, Material->data_name);
    strcat(filepath, ".txt");

    FILE *fp;
    fp = fopen(filepath, "r");

    if(fp == NULL)
    {
        printf("\033[40;31mFAME_Set_User_Define(19):\033[0m\n");
        printf("\033[40;31mThe material data '%s' is not found in %s.\033[0m\n", Material->data_name, filepath);
        printf("\033[40;31mPlease check material data name is correct in Popt.txt or the material data is in FAME_Material_Data.\033[0m\n");
        assert(0);
    }

    while(fgets(str, 100, fp) != NULL)
    {
        str[strlen(str) - 1] = '\0';
        if(strcmp(str, "# Data name") == 0)
        {
            fscanf(fp, "%s", Material->data_name);
        }

       else if(strcmp(str, "# Material number") == 0)
        {
            fscanf(fp, "%d", &Material->material_num);
            Material->sphere_num   = (int*) malloc(Material->material_num * sizeof(int));
            Material->cylinder_num = (int*) malloc(Material->material_num * sizeof(int));
        }

        else if(strcmp(str, "# Lattice type") == 0)
        {
            fscanf(fp, "%s", Lattice->lattice_type);
        }

        else if(strcmp(str, "# Lattice constant") == 0)
        {
            fscanf(fp, "%f %f %f %f %f %f", \
                   &Lattice->lattice_constant.a, &Lattice->lattice_constant.b, &Lattice->lattice_constant.c,\
                   &Lattice->lattice_constant.alpha, &Lattice->lattice_constant.beta, &Lattice->lattice_constant.gamma);
        }

        else if(strcmp(str, "# Sphere number") == 0)
        {
            for(i = 0; i < Material->material_num; i++)
            {
                fscanf(fp, "%d", &Material->sphere_num[i]);
                total_sphere += Material->sphere_num[i];
            }
            Material->sphere_centers = (realCPU*) malloc(total_sphere * 3 * sizeof(realCPU));
            Material->sphere_radius = (realCPU*) malloc(total_sphere * sizeof(realCPU));
        }

        else if(strcmp(str, "# Sphere centers") == 0)
        {
            for(i = 0; i < total_sphere * 3; i++)
                fscanf(fp, "%f", &Material->sphere_centers[i]);
        }

        else if(strcmp(str, "# Sphere radius") == 0)
        {
            if(Material->flag_radius_adjustment)
            {
                for(i = 0; i < total_sphere; i++)
                    Material->sphere_radius[i] = Material->flag_radius;
            }
            else
            {
                for(i = 0; i < total_sphere; i++)
                    fscanf(fp, "%f", &Material->sphere_radius[i]);
            }
        }

        else if(strcmp(str, "# Cylinder number") == 0)
        {
            total_cylinder = 0;
            for(i = 0; i < Material->material_num; i++)
            {
                fscanf(fp, "%d", &Material->cylinder_num[i]);
                total_cylinder += Material->cylinder_num[i];
            }
            Material->cylinder_top_centers = (realCPU*) malloc(total_cylinder * 3 * sizeof(realCPU));
            Material->cylinder_bot_centers = (realCPU*) malloc(total_cylinder * 3 * sizeof(realCPU));
            Material->cylinder_radius = (realCPU*) malloc(total_cylinder * sizeof(realCPU));
        }

        else if(strcmp(str, "# Cylinder top centers") == 0)
        {
            for(i = 0; i < total_cylinder * 3; i++)
                fscanf(fp, "%f", &Material->cylinder_top_centers[i]);
        }

        else if(strcmp(str, "# Cylinder bot centers") == 0)
        {
            for(i = 0; i < total_cylinder * 3; i++)
                fscanf(fp, "%f", &Material->cylinder_bot_centers[i]);
        }

        else if(strcmp(str, "# Cylinder radius") == 0)
        {
            if(Material->flag_radius_adjustment)
            {
                for(i = 0; i < total_cylinder; i++)
                    Material->cylinder_radius[i] = Material->flag_radius;
            }
            else
            {
                for(i = 0; i < total_cylinder; i++)
                    fscanf(fp, "%f", &Material->cylinder_radius[i]);
            }
        }
    }
    fclose(fp);
    return 0;
}