#ifndef _PRINTDEVICEARRAY_H_
#define _PRINTDEVICEARRAY_H_

void printDeviceArray(realGPU *d_Array, int n, const char *filename);

void printDeviceArray(cmpxGPU *d_Array, int n, const char *filename);

void printDeviceArray(int *d_Array, int n, const char *filename);

#endif
