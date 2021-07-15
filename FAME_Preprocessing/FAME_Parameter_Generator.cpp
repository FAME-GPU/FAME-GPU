#include "FAME_Internal_Common.h"
#include "FAME_Set_User_Define.h"
#include "FAME_Parameter_Lattice_Constants_Format.h"
#include "FAME_Parameter_Lattice_Vector.h"
#include "FAME_Parameter_Boundary_Point.h"
#include "FAME_Parameter_Recip_Lattice_Vector.h"

#include <iostream>
using namespace std;

void grid_nums_adjust(int* grid_nums, realCPU* edge_len, realCPU* mesh_len, int Nwant, int Nstep, int mem_size, int flag_Max);

int copy_Popt2Par(
    PAR*          Par, 
    FLAG          Popt_flag,
    MESH          Popt_mesh,
    MATERIAL      Popt_material,
    RECIP_LATTICE Popt_recip_lattice,
    LS            Popt_ls,
    ES            Popt_es);


int FAME_Parameter_Generator(
    PAR*          Par, 
    FLAG          Popt_flag,
    MESH          Popt_mesh,
    MATERIAL      Popt_material,
    RECIP_LATTICE Popt_recip_lattice,
    LS            Popt_ls,
    ES            Popt_es)
{	
    // Copy parameters from Popt to Par
    copy_Popt2Par(Par, Popt_flag, Popt_mesh, Popt_material, Popt_recip_lattice, Popt_ls, Popt_es);

    // Set material parameters to Par from desired material data file stated in Popt

	#if defined(USE_SINGLE)
	    FAME_Set_User_Define_Single(&Par->material, &Par->lattice);
	#else
	    FAME_Set_User_Define(&Par->material, &Par->lattice);
	#endif 

    // Set lattice constans to Par
    FAME_Parameter_Lattice_Constants_Format(&Par->lattice);

    // Set lattice vectors to Par
    FAME_Parameter_Lattice_Vector(&Par->lattice);

    // Compute edge length
    Par->mesh.edge_len[0] = Par->lattice.lattice_vec_a[0];
    Par->mesh.edge_len[1] = Par->lattice.lattice_vec_a[4];
    Par->mesh.edge_len[2] = Par->lattice.lattice_vec_a[8];

    // Let mesh_len[0] approximately equal mesh_len[1] approximately equal mesh_len[2]
    grid_nums_adjust(Par->mesh.grid_nums, Par->mesh.edge_len, Par->mesh.mesh_len, Par->es.nwant, Par->es.nstep, Par->flag.mem_size, Par->flag.grid_nums_max);
    printf("%*sGrid_nums = [ %3d %3d %3d ]\n", 54, "", Par->mesh.grid_nums[0], Par->mesh.grid_nums[1], Par->mesh.grid_nums[2]);

    // Set boundary point parameters to Par
    FAME_Parameter_Boundary_Point(&Par->lattice, Par->mesh);

    // Set reciprocal lattice vactors to Par
    FAME_Parameter_Recip_Lattice_Vector(Par->recip_lattice.reciprocal_lattice_vector_b, Par->lattice.lattice_vec_a_orig);

    return 0;
}

int copy_Popt2Par(
    PAR*          Par, 
    FLAG          Popt_flag,
    MESH          Popt_mesh,
    MATERIAL      Popt_material,
    RECIP_LATTICE Popt_recip_lattice,
    LS            Popt_ls,
    ES            Popt_es)
{
    // ======================== Mesh information ========================
    memcpy(Par->mesh.grid_nums, Popt_mesh.grid_nums, 3 * sizeof(int));
    // ================= Reciprocal lattice information =================
    Par->recip_lattice.part_num = Popt_recip_lattice.part_num;
    // ====================== Material information ======================
    strcpy((Par->material.data_name), (Popt_material.data_name));
    strcpy((Par->material.material_type), (Popt_material.material_type));

    Par->material.ele_permitt_in = (realCPU*) malloc(18 * sizeof(realCPU));
    memcpy(Par->material.ele_permitt_in, Popt_material.ele_permitt_in, 18 * sizeof(realCPU));

    Par->material.mag_permeab_in = (realCPU*) malloc(18 * sizeof(realCPU));
    memcpy(Par->material.mag_permeab_in, Popt_material.mag_permeab_in, 18 * sizeof(realCPU));

    Par->material.reciprocity_in = (realCPU*) malloc(18 * sizeof(realCPU));
    memcpy(Par->material.reciprocity_in, Popt_material.reciprocity_in, 9 * sizeof(realCPU));

    Par->material.chirality_in = (realCPU*) malloc(18 * sizeof(realCPU));
    memcpy(Par->material.chirality_in, Popt_material.chirality_in, 18 * sizeof(realCPU));

    Par->material.ele_permitt_out = Popt_material.ele_permitt_out;
    Par->material.mag_permeab_out = Popt_material.mag_permeab_out;
    Par->material.reciprocity_out = Popt_material.reciprocity_out;
    Par->material.chirality_out = Popt_material.chirality_out;
    // ======================= Solver information =======================
    Par->es.nwant = Popt_es.nwant;
    Par->es.nstep = Popt_es.nstep;
    Par->es.tol   = Popt_es.tol;
    Par->es.maxit = Popt_es.maxit;
    Par->ls.tol   = Popt_ls.tol;
    Par->ls.maxit = Popt_ls.maxit;
    // ========================== Flag setting ==========================
    Par->flag.device             = Popt_flag.device;
    Par->flag.printf_user_option = Popt_flag.printf_user_option;
    Par->flag.printf_parameter   = Popt_flag.printf_parameter;
    Par->flag.create_parameter   = Popt_flag.create_parameter;
    Par->flag.create_B_inout     = Popt_flag.create_B_inout;
    Par->flag.create_wave_vector = Popt_flag.create_wave_vector;
    Par->flag.save_eigen_vector  = Popt_flag.save_eigen_vector;
    Par->flag.grid_nums_max      = Popt_flag.grid_nums_max;
    Par->flag.mem_size           = Popt_flag.mem_size;
    Par->flag.radius_adjustment  = Popt_flag.radius_adjustment;
    Par->flag.radius             = Popt_flag.radius;
    Par->material.flag_radius_adjustment = Popt_flag.radius_adjustment;
    Par->material.flag_radius    = Popt_flag.radius;

    return 0;
}

void grid_nums_adjust(int* grid_nums, realCPU* edge_len, realCPU* mesh_len, int Nwant, int Nstep, int mem_size, int flag_Max)
{   
    mesh_len[0] = edge_len[0] / grid_nums[0];
    grid_nums[1] = (int) round( edge_len[1] / mesh_len[0] );
    mesh_len[1] = edge_len[1] / grid_nums[1];
    grid_nums[2] = (int) round( edge_len[2] / mesh_len[0] );
    mesh_len[2] = edge_len[2] / grid_nums[2];
    
    if(flag_Max)
    {
        size_t byte = 768 + 32 * (Nwant + Nstep + 1) + 128;
        size_t a = floor(mem_size * 1024 * 1024 / (realCPU) byte);
        size_t b = floor(Nstep * Nstep * 16 / (realCPU) byte);
        size_t maxsize = a * 1024 - b;
        int Nx, Ny, Nz, flag = 0;

        while(1)
        {
            Nx = grid_nums[0];
            Ny = grid_nums[1];
            Nz = grid_nums[2];
            if(Nx * Ny * Nz <= maxsize)
            {
                if(flag)
                    break;
                grid_nums[0] += 4;
                mesh_len[0] = edge_len[0] / grid_nums[0];
                grid_nums[1] = (int) round( edge_len[1] / mesh_len[0] );
                mesh_len[1] = edge_len[1] / grid_nums[1];
                grid_nums[2] = (int) round( edge_len[2] / mesh_len[0] );
                mesh_len[2] = edge_len[2] / grid_nums[2];
            }
            else
            {
                grid_nums[0] -= 4;
                mesh_len[0] = edge_len[0] / grid_nums[0];
                grid_nums[1] = (int) round( edge_len[1] / mesh_len[0] );
                mesh_len[1] = edge_len[1] / grid_nums[1];
                grid_nums[2] = (int) round( edge_len[2] / mesh_len[0] );
                mesh_len[2] = edge_len[2] / grid_nums[2];
                flag = 1;
            }
        }
    }
}