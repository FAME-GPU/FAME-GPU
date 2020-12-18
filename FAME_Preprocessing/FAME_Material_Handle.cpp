#include "FAME_Internal_Common.h"
// 2020-02-19

int sphere_handle(realCPU ptx, realCPU pty, realCPU ptz, realCPU* lattice_vec_a, MATERIAL Material, int kind)
{
    int i, sum = 0, flag = 0;
    int shiftx, shifty, shiftz;
    realCPU center_x, center_y, center_z;
    realCPU distance;

    for(i = 0; i < kind; i++)
        sum += Material.sphere_num[i];

    for(i = sum; i < sum + Material.sphere_num[kind]; i++)
        for(shiftx = -1; shiftx <= 1; shiftx++)
            for(shifty = -1; shifty <= 1; shifty++)
                for(shiftz = -1; shiftz <= 1; shiftz++)
                {
                    center_x = (Material.sphere_centers[i * 3] + shiftx) * lattice_vec_a[0] + (Material.sphere_centers[i * 3 + 1] + shifty) * lattice_vec_a[3] + (Material.sphere_centers[i * 3 + 2] + shiftz) * lattice_vec_a[6];
                    center_y = (Material.sphere_centers[i * 3] + shiftx) * lattice_vec_a[1] + (Material.sphere_centers[i * 3 + 1] + shifty) * lattice_vec_a[4] + (Material.sphere_centers[i * 3 + 2] + shiftz) * lattice_vec_a[7];
                    center_z = (Material.sphere_centers[i * 3] + shiftx) * lattice_vec_a[2] + (Material.sphere_centers[i * 3 + 1] + shifty) * lattice_vec_a[5] + (Material.sphere_centers[i * 3 + 2] + shiftz) * lattice_vec_a[8];

                    distance = pow(center_x - ptx, 2) + pow(center_y - pty, 2) + pow(center_z - ptz, 2);

                    if(distance <= pow(Material.sphere_radius[i], 2))
                    {
                        flag = 1;
                        break;
                    }
                }
    return flag;
}

int cylinder_handle(realCPU ptx, realCPU pty, realCPU ptz, realCPU* lattice_vec_a, MATERIAL Material, int kind)
{
    int i, sum = 0, flag = 0;
    int shiftx, shifty, shiftz;
    realCPU ctx, cty, ctz, cbx, cby, cbz;
    realCPU temp, distance;

    for(i = 0; i < kind; i++)
        sum += Material.cylinder_num[i];

    for(i = sum; i < sum + Material.cylinder_num[kind]; i++)
    {
        for(shiftx = -1; shiftx <= 1; shiftx++)
            for(shifty = -1; shifty <= 1; shifty++)
                for(shiftz = -1; shiftz <= 1; shiftz++)
                {
                    ctx = (Material.cylinder_top_centers[i * 3] + shiftx) * lattice_vec_a[0] + (Material.cylinder_top_centers[i * 3 + 1] + shifty) * lattice_vec_a[3] + (Material.cylinder_top_centers[i * 3 + 2] + shiftz) * lattice_vec_a[6];
                    cty = (Material.cylinder_top_centers[i * 3] + shiftx) * lattice_vec_a[1] + (Material.cylinder_top_centers[i * 3 + 1] + shifty) * lattice_vec_a[4] + (Material.cylinder_top_centers[i * 3 + 2] + shiftz) * lattice_vec_a[7];
                    ctz = (Material.cylinder_top_centers[i * 3] + shiftx) * lattice_vec_a[2] + (Material.cylinder_top_centers[i * 3 + 1] + shifty) * lattice_vec_a[5] + (Material.cylinder_top_centers[i * 3 + 2] + shiftz) * lattice_vec_a[8];
                    cbx = (Material.cylinder_bot_centers[i * 3] + shiftx) * lattice_vec_a[0] + (Material.cylinder_bot_centers[i * 3 + 1] + shifty) * lattice_vec_a[3] + (Material.cylinder_bot_centers[i * 3 + 2] + shiftz) * lattice_vec_a[6];
                    cby = (Material.cylinder_bot_centers[i * 3] + shiftx) * lattice_vec_a[1] + (Material.cylinder_bot_centers[i * 3 + 1] + shifty) * lattice_vec_a[4] + (Material.cylinder_bot_centers[i * 3 + 2] + shiftz) * lattice_vec_a[7];
                    cbz = (Material.cylinder_bot_centers[i * 3] + shiftx) * lattice_vec_a[2] + (Material.cylinder_bot_centers[i * 3 + 1] + shifty) * lattice_vec_a[5] + (Material.cylinder_bot_centers[i * 3 + 2] + shiftz) * lattice_vec_a[8];

                    temp = ((ptx - ctx) * (cbx - ctx) + (pty - cty) * (cby - cty) + (ptz - ctz) * (cbz - ctz)) / (pow(cbx - ctx, 2) + pow(cby - cty, 2) + pow(cbz - ctz, 2));

                    if(0.0 <= temp && temp <= 1.0)
                    {
                        distance = pow(ctx + temp * (cbx - ctx) - ptx, 2) + pow(cty + temp * (cby - cty) - pty, 2) + pow(ctz + temp * (cbz - ctz) - ptz, 2);
                        
                        if(distance <= pow(Material.cylinder_radius[i], 2))
                        {
                            flag = 1;
                            break;
                        }
                    }
                }
    }
    return flag;
}

int FAME_Material_Handle(realCPU* point_set, realCPU* lattice_vec_a, PAR Par)
{
    int i;
    int flag_sphere, flag_cylinder;

    for(i = 0; i < Par.material.material_num; i++)
    {
        flag_sphere   = sphere_handle(point_set[0], point_set[1], point_set[2], lattice_vec_a, Par.material, i);
        flag_cylinder = cylinder_handle(point_set[0], point_set[1], point_set[2], lattice_vec_a, Par.material, i);
        if(flag_sphere || flag_cylinder)
            return i;
    }

    return -1;
}