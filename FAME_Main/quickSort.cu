#include "FAME_Internal_Common.h"
#include "FAME_CUDA.h"


int partition(realGPU* arr, int *idx, int left, int right);
void swap(realGPU *arr, int *idx, int i, int j);

void quickSort ( realGPU* A, int* idx, int left, int right ) 
{ 
    int partitionIndex;
    if (left < right) 
    {
        partitionIndex = partition(A, idx, left, right);
        quickSort(A, idx, left, partitionIndex-1);
        quickSort(A, idx, partitionIndex+1, right);
    }
}
 
int partition ( realGPU* arr, int *idx, int left, int right )
{     
    int pivot = left,                      
        index = pivot + 1;
    for (int i = index; i <= right; i++) 
    {
        if (fabs(arr[i]) > fabs(arr[pivot])) 
        {
            swap(arr, idx, i, index);
            index++;
        }       
    }
    swap(arr, idx, pivot, index - 1);
    return index-1;
}
 
void swap ( realGPU *arr, int *idx, int i, int j ) 
{
    realGPU temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
    int tep = idx[i];
    idx[i] = idx[j];
    idx[j] = tep;
}