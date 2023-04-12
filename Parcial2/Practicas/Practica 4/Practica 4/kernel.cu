#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <random>
#include <iostream>
using namespace std;

const int s = 5;

struct AoS {
    int value;
};

struct SoA
{
    int values[s];
};

__global__ void arrayOfStructuresFunction(AoS* a)
{
    printf("AoS [%d] : %d \n", threadIdx.x, a[threadIdx.x].value);
}

__global__ void structureOfArrayFunction(SoA* a)
{
    printf("SoA[%d] : % d \n", threadIdx.x, a[0].values[threadIdx.x]);
}

int main()
{
    AoS arrayOfStructures_Host[s];
    SoA structureOfArray_Host[1];
    AoS* arrayOfStructures_Device;
    SoA* structureOfArray_Device;
    cudaMalloc(&arrayOfStructures_Device, s * sizeof(AoS));
    cudaMalloc(&structureOfArray_Device, sizeof(SoA));

    for (int i = 0; i < s; i++)
    {
        int value = rand() % 10;
        arrayOfStructures_Host[i].value = value;
        structureOfArray_Host[0].values[i] = value;
    }

    cudaMemcpy(arrayOfStructures_Device, arrayOfStructures_Host, s * sizeof(AoS), cudaMemcpyHostToDevice);
    cudaMemcpy(structureOfArray_Device, structureOfArray_Host, sizeof(SoA), cudaMemcpyHostToDevice);

    arrayOfStructuresFunction << <1, 5 >> > (arrayOfStructures_Device);
    structureOfArrayFunction << <1, 5 >> > (structureOfArray_Device);

    cudaFree(arrayOfStructures_Device);
    cudaFree(structureOfArray_Device);

    return 0;
}