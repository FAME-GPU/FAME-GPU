#include "FAME_Internal_Common.h"
#include "FAME_Preprocessing.h"
#include "FAME_Main_Code.cuh"
#include "FAME_Profile.h"

void FAME_Free(PAR Par, PROFILE Profile);

int main()
{
    PAR Par;
    PROFILE Profile;
    realCPU accum_Pre, accum_Mai;
    struct timespec start, end;

    printf("\033[40;33m= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\033[0m\n");
	printf("\033[40;33m= = = = = = = = = = = = = = = = Start to run FAME = = = = = = = = = = = = = = = =\033[0m\n");
    printf("\033[40;33m= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\033[0m\n");

    printf("= = = = FAME_Preprocessing  = = = = = = = = = = = = = = = = = = = = = = = = = = =\n");
    clock_gettime(CLOCK_REALTIME, &start);
    FAME_Preprocessing(&Par);
    clock_gettime(CLOCK_REALTIME, &end);
    accum_Pre = ( end.tv_sec - start.tv_sec ) + ( end.tv_nsec - start.tv_nsec ) / BILLION;
    //printf("%*s%8.2f sec.\n", 68, "", accum_Pre);

    FAME_Set_Profile(&Profile, Par);

    printf("\n= = FAME_Main_Code  = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n");

    clock_gettime(CLOCK_REALTIME, &start);
    FAME_Main_Code(Par, &Profile);
    clock_gettime(CLOCK_REALTIME, &end);
    accum_Mai = ( end.tv_sec - start.tv_sec ) + ( end.tv_nsec - start.tv_nsec ) / BILLION;

    printf("FAME_Preprocessing%*s%8.2f sec.\n", 50, "", accum_Pre);
    printf("FAME_Main_Code%*s%12.2f sec.\n", 50, "", accum_Mai);
    printf("TOTAL%*s%12.2f sec.\n", 59, "", accum_Pre + accum_Mai);
    
    FAME_Create_Profile_txt(Profile);

    FAME_Free(Par, Profile);

    printf("\033[40;33m= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\033[0m\n");
    if(strcmp(Par.material.material_type, "isotropic") == 0)
	{
		printf("\033[40;33m= = = = = = = = = = = = FAME (Isotropic GPU) has done ! = = = = = = = = = = = = =\033[0m\n");
	}
    if(strcmp(Par.material.material_type, "anisotropic") == 0)
	{
		printf("\033[40;33m= = = = = = = = = = = = FAME (Anisotropic GPU) has done ! = = = = = = = = = = = = =\033[0m\n");
	}
	else if(strcmp(Par.material.material_type, "biisotropic") == 0)
	{
        printf("\033[40;33m= = = = = = = = = = = = FAME (Biisotropic GPU) has done ! = = = = = = = = = = = = =\033[0m\n");
    }   
    if(strcmp(Par.material.material_type, "bianisotropic") == 0)
	{
		printf("\033[40;33m= = = = = = = = = = = = FAME (Bianisotropic GPU) has done ! = = = = = = = = = = = = =\033[0m\n");
	}
    printf("\033[40;33m= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\033[0m\n");

    return 0;
}

void FAME_Free(PAR Par, PROFILE Profile)
{
    free(Par.lattice.lattice_constant.t1);
    free(Par.lattice.lattice_constant.t2);
    free(Par.lattice.lattice_constant.t3);
    free(Par.lattice.lattice_constant.t4);

    free(Par.material.sphere_num);
    free(Par.material.sphere_centers);
    free(Par.material.sphere_radius);
    free(Par.material.cylinder_num);
    free(Par.material.cylinder_top_centers);
    free(Par.material.cylinder_bot_centers);
    free(Par.material.cylinder_radius);

    free(Par.recip_lattice.WaveVector);

    FAME_Free_Profile(Profile);
}
