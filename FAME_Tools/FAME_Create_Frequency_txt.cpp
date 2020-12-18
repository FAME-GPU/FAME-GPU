#include "FAME_Internal_Common.h"

int FAME_Create_Frequency_txt(realCPU* Freq_array, int Nwant, int idx)
{
	int i, j;
	FILE* fp;
	fp = fopen("Data_Omega_array.txt", "w");

	//fprintf(fp, "Freq_array = []\n");
	for(i = 0; i <= idx; i++)
	{
		//fprintf(fp, "Freq_array(:, %d) = [\n", i + 1);
		for(j = 0; j < Nwant; j++)
			fprintf(fp, "% 10.12lf\n", Freq_array[j + i * Nwant]);
		//fprintf(fp, "];\n");
	}
	
	fclose(fp);

	return 0;
}