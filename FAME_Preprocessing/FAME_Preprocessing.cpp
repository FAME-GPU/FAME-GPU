#include "FAME_Internal_Common.h"
#include "FAME_Set_User_Option.h"
#include "FAME_Parameter_Generator.h"
#include "FAME_Parameter_Brillouin_Zone_Path.h"
#include "FAME_Material_Locate_Index.h"
#include "FAME_Print_User_Option.h"
#include "FAME_Print_Parameter.h"
#include "FAME_Create_Par_txt.h"
#include "FAME_Create_B_txt.h"
#include "FAME_Create_WaveVector_txt.h"


int FAME_Preprocessing(PAR* Par)
{
    POPT Popt;
    struct timespec start, end;
    int single;
	
    printf("= = = = FAME_Set_User_Option  = = = = = = = = = = = = = = = = = = = = = = = = = =\n");

    #if defined(USE_SINGLE)
        FAME_Set_User_Option_Single(&Popt);
    #else
        FAME_Set_User_Option(&Popt);
    #endif 
        
    
    printf("= = = = FAME_Parameter_Generator  = = = = = = = = = = = = = = = = = = = = = = = =\n");
    FAME_Parameter_Generator(Par, Popt.flag, Popt.mesh, Popt.material, Popt.recip_lattice, Popt.ls, Popt.es);

    printf("= = = = FAME_Parameter_Brillouin_Zone_Path  = = = = = = = = = = = = = = = = = = =\n");
    FAME_Parameter_Brillouin_Zone_Path(&Par->recip_lattice, Par->recip_lattice.part_num, Par->lattice);

    printf("= = = = FAME_Material_Locate_Index  = = = = = = = = = = = = = = = = = = = = = = =\n");
    clock_gettime(CLOCK_REALTIME, &start);
    FAME_Material_Locate_Index(&Par->material, *Par);
    clock_gettime(CLOCK_REALTIME, &end);
    realCPU accum = ( end.tv_sec - start.tv_sec ) + ( end.tv_nsec - start.tv_nsec ) / BILLION;
    printf("%*s%8.2f sec.\n", 68, "", accum);


//待改
    if(Par->flag.printf_user_option)
    {
        printf("= = = = FAME_Print_User_Option  = = = = = = = = = = = = = = = = = = = = = = = = =\n");
        FAME_Print_User_Option(Popt);
    }

    if(Par->flag.printf_parameter)
    {
        printf("= = = = FAME_Print_Parameter  = = = = = = = = = = = = = = = = = = = = = = = = = =\n");
        FAME_Print_Parameter(*Par);
    }

    if(Par->flag.create_parameter)
    {
        printf("= = = = FAME_Create_Par_txt = = = = = = = = = = = = = = = = = = = = = = = = = = =\n");
        FAME_Create_Par_txt(*Par);
    }

    if(Par->flag.create_B_inout)
    {
        printf("= = = = FAME_Create_B_txt = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n");
        FAME_Create_B_txt(Par->material, Par->mesh.grid_nums);
    }

    if(Par->flag.create_wave_vector)
    {
        printf("= = = = FAME_Create_WaveVector_txt  = = = = = = = = = = = = = = = = = = = = = = =\n");
        FAME_Create_WaveVector_txt(Par->recip_lattice.Wave_vec_num, Par->recip_lattice.WaveVector);
    }

    free(Popt.material.ele_permitt_in);
    return 0;
}
