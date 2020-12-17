#include "FAME_Internal_Common.h"

int FAME_Set_Profile(PROFILE* Profile, PAR Par)
{
    Profile->ls_iter = (int*)    calloc(Par.recip_lattice.Wave_vec_num, sizeof(int));
    Profile->es_iter = (int*)    calloc(Par.recip_lattice.Wave_vec_num, sizeof(int));
    Profile->ls_time = (double*) calloc(Par.recip_lattice.Wave_vec_num, sizeof(double));
    Profile->es_time = (double*) calloc(Par.recip_lattice.Wave_vec_num, sizeof(double));
    return 0;
}

int FAME_Print_Profile(PROFILE Profile)
{
	printf("= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n");
	printf("               ");
	printf("idx     LS_iter     LS_time     ES_iter     ES_time\n");
	printf("               ");
    printf("%3d",      Profile.idx + 1);
	printf("%12d",     Profile.ls_iter[Profile.idx]);
	printf("%12.2f",   Profile.ls_time[Profile.idx]);
	printf("%12d",     Profile.es_iter[Profile.idx]);
	printf("%12.2f\n", Profile.es_time[Profile.idx]);
    printf("               ");
	printf("= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n");
  	return 0;
}

int FAME_Create_Profile_txt(PROFILE Profile)
{
    FILE* fp;
    fp = fopen("Data_Profile.txt", "w");
    fprintf(fp, "Compute_Info = [\n");
    fprintf(fp, "%%idx   LS_iter     LS_time   ES_iter     ES_time\n");
    for(int i = 0; i <= Profile.idx; i++)
        fprintf(fp, " %3d, %8d, %10.2f, %8d, %10.2f;\n", i + 1, Profile.ls_iter[i], Profile.ls_time[i], Profile.es_iter[i], Profile.es_time[i]);
    fprintf(fp, "];\n");
    fclose(fp);

    return 0;
}

int FAME_Free_Profile(PROFILE Profile)
{
    free(Profile.ls_iter);
    free(Profile.es_iter);
    free(Profile.ls_time);
    free(Profile.es_time);
  	return 0;
}
