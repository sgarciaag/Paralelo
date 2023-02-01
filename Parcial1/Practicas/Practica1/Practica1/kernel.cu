#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

__global__ void Suma(int* a, int* b, int* c)
{
    /* printf("threadx - x: %d, y: %d, z: %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
     printf("blockidx - x: %d, y: %d, z: %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
     printf("gridDim - x: %d, y: %d, z: %d\n", gridDim.x, gridDim.y, gridDim.z);*/

    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
    printf("%d\n", c[threadIdx.x]);

}

// nothing == host - cpu 
// d_ == global - gpu

int main() {

    const int n = 5;

    //Host CPU variables
    int a[n] = { 1,2,3,4,5 };
    int b[n] = { 1,2,3,4,5 };
    int c[n] = { 0 };

    int size = sizeof(int) * n;

    //GPU variables d_
    int* d_a;
    int* d_b;
    int* d_c;

    // malloc Cuda
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    //Cuda Memcopy host to device: d_c <- c
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);


    dim3 grid(1);
    dim3 block(n,1,1);


    Suma << < grid, block >> > (d_a, d_b, d_c);
    cudaDeviceSynchronize();

    //Cuda Memcopy device to host: c <- d_c
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    printf("\nArreglos sumados: ");
    printf("%d, %d, %d, %d, %d", c[0], c[1], c[2], c[3], c[4]);

    //Cuda free
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}