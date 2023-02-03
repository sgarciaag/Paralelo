#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>


using namespace std;

// Errores

__host__ void check_CUDA_error(const char* e) {
    cudaError_t error;
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("\nERROR %d: %s (%s)", error, cudaGetErrorString(error), e);
    }
}

__global__ void idx_calc_tid(int* a)
{

    int tid = threadIdx.x;

    printf("threadx - x: %d, data: %d\n", tid, a[tid]);
    /* printf("blockidx - x: %d, y: %d, z: %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("gridDim - x: %d, y: %d, z: %d\n", gridDim.x, gridDim.y, gridDim.z);*/


}

__global__ void idx_calc_gid(int* a)
{

    int tid = threadIdx.x;
    //gid = tid + offset;
    int offset = blockIdx.x * blockDim.x;
    int gid = tid + offset;


    printf("blockIdx - x: %d, threadx - x: %d, gid: %d, data: %d\n", blockIdx.x, tid, gid, a[gid]);
    /* printf("blockidx - x: %d, y: %d, z: %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("gridDim - x: %d, y: %d, z: %d\n", gridDim.x, gridDim.y, gridDim.z);*/


}

__global__ void idx_calc_gid2D(int* a)
{

    int tid = threadIdx.x;
    //gid = tid + offset;
    int offsetRow = gridDim.x * blockDim.x * blockIdx.y;
    int offsetBlock = blockIdx.x * blockDim.x;
    int gid = tid + offsetBlock + offsetRow;


    printf("blockIdx - x: %d, threadx - x: %d, gid: %d, data: %d\n", blockIdx.x, tid, gid, a[gid]);
    /* printf("blockidx - x: %d, y: %d, z: %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("gridDim - x: %d, y: %d, z: %d\n", gridDim.x, gridDim.y, gridDim.z);*/


}

__global__ void idx_calc_gid2D2(int* a)
{

    int tid = threadIdx.x;
    int n_thBlock = blockDim.x * blockDim.y;
    int n_thRow = n_thBlock * gridDim.x;
    //gid = tid + offset;
    int offsetRow = n_thRow * blockIdx.y;
    int offsetBlock = blockIdx.x * n_thBlock;
    int gid = tid + offsetBlock + offsetRow;


    printf("blockIdx - x: %d, threadx - x: %d, gid: %d, data: %d\n", blockIdx.x, tid, gid, a[gid]);
    /* printf("blockidx - x: %d, y: %d, z: %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("gridDim - x: %d, y: %d, z: %d\n", gridDim.x, gridDim.y, gridDim.z);*/


}

__global__ void idx_calc_gid3D(int* input) {

    int totalThreads = blockDim.x * blockDim.y * blockDim.z;

    int tid = 
        threadIdx.x //1D
        + threadIdx.y * blockDim.x //2D
        + threadIdx.z * blockDim.x * blockDim.z; //3D

    int bid =
        blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + blockIdx.z * gridDim.x * gridDim.y; //3D

    int gid = tid + bid * totalThreads; // thread ID + offset

    printf("[DEVICE] gid: %d, data: %d\n\r", gid, input[gid]);
}

__global__ void sum_array_gpu(int* a, int* b, int* c, int size) {

    int totalThreads = blockDim.x * blockDim.y * blockDim.z;

    int tid = threadIdx.x //ID
        + threadIdx.y * blockDim.x //2D
        + threadIdx.z * blockDim.x * blockDim.y; //3D

    int bid = blockIdx.x //ID
        + blockIdx.y * gridDim.x //2D
        + blockIdx.z * gridDim.x * gridDim.y; //3D

    int gid = tid + bid * totalThreads; //threadIdx + offset
    if (gid < size)
    {
        c[gid] = a[gid] + b[gid];
        printf("[DEVICE] gid: %d, %d \n", gid, c[gid]);
    }

}


// h - host - cpu 
// g - global - gpu

// _ - host
// d_ - global
//

int main() {

    //BEGIN funciones normales
    const int n = 16;

    int a[n] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    int b[n] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    int c[n] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    
    int size = sizeof(int) * n;

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

    //END funciones normales

    dim3 gridTid(1);
    dim3 blockTid(n, 1, 1);

    printf("TID\n");

    idx_calc_tid << <gridTid, blockTid >> > (d_a);
    check_CUDA_error("Error en lanzamiento del kernel");
    cudaDeviceSynchronize();


    dim3 gridGid(4, 1, 1);
    dim3 blockGid(4, 1, 1);


    printf("GID\n");

    idx_calc_gid << <gridGid, blockGid >> > (d_a);
    check_CUDA_error("Error en lanzamiento del kernel");
    cudaDeviceSynchronize();

    dim3 gridGid2D(2, 2, 1);
    dim3 blockGid2D(4, 1, 1);


    printf("GID 2D\n");

    idx_calc_gid2D << <gridGid2D, blockGid2D >> > (d_a);
    check_CUDA_error("Error en lanzamiento del kernel");
    cudaDeviceSynchronize();

    dim3 gridGid2D2(2, 2, 1);
    dim3 blockGid2D2(2, 2, 1);


    printf("GID 2D2\n");

    idx_calc_gid2D2 << <gridGid2D2, blockGid2D2 >> > (d_a);
    check_CUDA_error("Error en lanzamiento del kernel");
    cudaDeviceSynchronize();


    dim3 gridGid3D(2, 2, 2);
    dim3 blockGid3D(2, 2, 2);

    printf("GID 2D2\n");

    idx_calc_gid3D << <gridGid3D, blockGid3D >> > (d_a);
    check_CUDA_error("Error en lanzamiento del kernel");
    cudaDeviceSynchronize();

    //BEGIN funcion suma
    const int nSuma = 10000;
    int sizeSuma = sizeof(int) * nSuma;

    int* aSuma = (int*)malloc(nSuma);
    int* bSuma = (int*)malloc(nSuma);
    int* cSuma = (int*)malloc(nSuma);
    int* c_gpu_result = (int*)malloc(nSuma);

    for (int i = 0; i < nSuma; i++) {
        aSuma[i] = i;
        bSuma[i] = i;

    }

    int* d_aSuma;
    int* d_bSuma;
    int* d_cSuma;

    // malloc Cuda
    cudaMalloc(&d_aSuma, sizeSuma);
    cudaMalloc(&d_bSuma, sizeSuma);
    cudaMalloc(&d_cSuma, sizeSuma);

    //Cuda Memcopy host to device: d_c <- c

    cudaMemcpy(d_aSuma, aSuma, sizeSuma, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bSuma, bSuma, sizeSuma, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cSuma, cSuma, sizeSuma, cudaMemcpyHostToDevice);

    //END funcion suma

    dim3 gridGid3DSum(10,10);
    dim3 blockGid3DSum(32, 2, 2);

    clock_t gpu_start, gpu_stop;

    printf("\nArray Sum:\n");
    gpu_start = clock();
    sum_array_gpu << <gridGid3DSum, blockGid3DSum >> > (d_aSuma, d_bSuma, d_cSuma, nSuma);
    check_CUDA_error("Error en el lanzamiento del kernel");
    cudaDeviceSynchronize();
    gpu_stop = clock();

    double cps_gpu = (double)((double)(gpu_stop - gpu_start) / CLOCKS_PER_SEC);
    printf("Execution Time [ET-GPU]: %4.6f\n\r", cps_gpu);

    //Cuda free
    cudaFree(d_a);
    cudaDeviceReset();

    return 0;
}