#ifndef _PRINTDEVICEARRAY_H_
#define _PRINTDEVICEARRAY_H_

void printDeviceArray(double *d_Array, int n, const char *filename);

void printDeviceArray(cuDoubleComplex *d_Array, int n, const char *filename);

void printDeviceArray(int *d_Array, int n, const char *filename);

void printDeviceArray(cmpx *d_Array, int n, const char *filename);
#endif
