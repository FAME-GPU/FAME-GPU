#include "FAME_Internal_Common.h"
// 2020-02-19

int FAME_Create_WaveVector_txt(int WaveVecotrNum, double* WaveVector)
{
    int i;
    FILE* fp;
    fp = fopen("Data_WaveVector.txt", "w");

    for(i = 0; i < WaveVecotrNum; i++)
        fprintf(fp, "% lf % lf % lf\n", WaveVector[i * 3 + 0], WaveVector[i * 3 + 1], WaveVector[i * 3 + 2]);

    fclose(fp);

    return 0;
}
