﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void sumArray(int* a, int* b, int* c, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) c[i] = a[i] + b[i];
}

__global__ void sumArrayZero(int* a, int* b, int* c, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) c[i] = a[i] + b[i];
}

__global__ void unrolling2(int* input, int* temp, int size)
{
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x * 2;
    int index = block_offset + tid;
    int* i_data = input + block_offset;
    if ((index + blockDim.x) < size)
    {
        input[index] += input[index + blockDim.x];
    }

    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset = offset / 2)
    {
        if (tid < offset)
        {
            i_data[tid] += i_data[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        temp[blockIdx.x] = i_data[0];
    }
}

__global__ void unrolling4(int* input, int* temp, int size)
{
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x * 4;
    int index = block_offset + tid;
    int* i_data = input + block_offset;
    if ((index + 3 * blockDim.x) < size)
    {
        int a1 = input[index];
        int a2 = input[index + blockDim.x];
        int a3 = input[index + 2 * blockDim.x];
        int a4 = input[index + 3 * blockDim.x];
        input[index] += a1 + a2 + a3 + a4;
    }


    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset = offset / 2)
    {
        if (tid < offset)
        {
            i_data[tid] += i_data[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        temp[blockIdx.x] = i_data[0];
    }
}

__global__ void unrolling_warps(int* input, int* temp, int size)
{
    int tid = threadIdx.x;
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int* i_data = input + blockDim.x * blockIdx.x;

    for (int offset = blockDim.x / 2; offset >= 64; offset = offset / 2)
    {
        if (tid < offset)
        {
            i_data[tid] += i_data[tid + offset];
        }

        __syncthreads();
    }
    if (tid < 32)
    {
        volatile int* vsmem = i_data;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }
    if (tid == 0)
    {
        input[blockIdx.x] = i_data[0];
    }
}
__global__ void unrolling_complete(int* input, int* temp, int size)
{

}

int main()
{

    // get Device Properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    // check if support mapped memory
    if (!deviceProp.canMapHostMemory)
    {
        printf("Device %d does not support mapping CPU host memory!\n", 0);
        cudaDeviceReset();
        return 0;
    }

    //int data_size = 1 << 22;
    int data_size = 4096;
    size_t byte_size = data_size * sizeof(int);

    // malloc host memory
    int* h_a, * h_b, * h_ref, * h_g_ref;
    h_a = (int*)malloc(byte_size);
    h_b = (int*)malloc(byte_size);
    h_ref = (int*)malloc(byte_size);
    h_g_ref = (int*)malloc(byte_size);

    srand((unsigned)time(NULL));
    for (int i = 0; i < data_size; i++)
    {
        h_a[i] = 1;
    }
    int* d_input, * d_temp;
    cudaMalloc((void**)&d_input, byte_size);
    cudaMalloc((void**)&d_temp, byte_size);

    cudaMemset(d_temp, 0, byte_size);
    cudaMemcpy(d_input, h_a, byte_size, cudaMemcpyHostToDevice);

    dim3 block(128);
    dim3 grid((data_size / 128) / 2);
    unrolling2 << <grid, block >> > (d_input, d_temp, data_size);

    cudaMemcpy(h_b, d_input, byte_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 100; i++)
    {
        printf("%d", h_b[i]);
    }

    //reset device
    cudaDeviceReset();
    return 0;

}