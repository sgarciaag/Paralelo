#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

__global__ void sort_kernel(int* d_array, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                if (d_array[i] > d_array[j])
                {
                    int temp = d_array[i];
                    d_array[i] = d_array[j];
                    d_array[j] = temp;
                }
            }
        }
    }
}

int main()
{
    const int N = 20;
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int h_array[N] = { 5, 2, 1, 7, 3, 9, 8, 4, 6, 0, 5, 2, 1, 7, 3, 9, 8, 4, 6, 0 };
    int* d_array;

    printf("Original array: ");
    for (int i = 0; i < N; i++)
    {
        printf("%d ", h_array[i]);
    }
    printf("\n");
    
    cudaMalloc((void**)&d_array, N * sizeof(int));
    cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice);
    sort_kernel << <blocksPerGrid, threadsPerBlock >> > (d_array, N);
    cudaMemcpy(h_array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_array);
    printf("Sorted GPU array: ");
    for (int i = 0; i < N; i++)
    {
        printf("%d ", h_array[i]);
    }
    printf("\n");
    return 0;
}
