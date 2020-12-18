#include "FAME_Internal_Common.h"
// 2020-02-19

int FAME_Create_B_txt(MATERIAL Material, int* grid_nums)
{
    int i, j, kind;
    int idx = Material.material_num * grid_nums[0] * grid_nums[1] * grid_nums[2];

    if(strcmp(Material.material_type, "isotropic") == 0)
        kind = 3;

    FILE* fp;
    fp = fopen("Data_B.txt", "w");

    for (i = 0; i < kind; i++)
    {
        for (j = 0; j < idx; j++)
            fprintf(fp, "%d ", Material.Binout[j + i * idx]);
        fprintf(fp, "\n\n");
    }

    fclose(fp);

    return 0;
}
