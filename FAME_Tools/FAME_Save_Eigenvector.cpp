#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
typedef double _Complex cmpx;

int FAME_Save_Eigenvector(cmpx* Ele_field_mtx, int Nwant, int N3, int idx)
{
	FILE* fp;
	int zero = 0;
	char temp[100];
	char name[100] = "Data_EV_";
	char filename[100];

	for(int i = 0; i < Nwant; i++)
	{
		strcpy(filename, name);
		if(idx + 1 < 10)
		{
			sprintf(temp, "%d", 0);
			strcat(filename, temp);
		}

		sprintf(temp, "%d", idx + 1);
		strcat(filename, temp);
		strcat(filename, "_");

		if(i + 1 < 10)
		{
			sprintf(temp, "%d", 0);
			strcat(filename, temp);
		}

		sprintf(temp, "%d", i + 1);
		strcat(filename, temp);
		strcat(filename, ".txt");
		fp = fopen(filename, "w");

		fprintf(fp, "Ele_field_mtx{%d}(:, %d) = [\n", idx + 1, i + 1);
		for(int j = 0; j < N3; j++)
			fprintf(fp, "(% e) + (% e * 1i)\n", creal(Ele_field_mtx[i * N3 + j]), cimag(Ele_field_mtx[i * N3 + j]));
		fprintf(fp, "];\n", idx + 1, i + 1);
		fclose(fp);
	}

	return 0;
}