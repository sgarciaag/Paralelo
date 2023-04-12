#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cuda_runtime_api.h>

//Kernel
__global__ void threshold_kernel(unsigned char* input, unsigned char* output, int width, int height, int threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;
        output[idx] = (input[idx] >= threshold) ? 1 : 0;
    }
}


int main()
{
    // Definir el tamaño de la imagen
    int N = 1024;
    int M = 768;

    // Leer la imagen de entrada
    // Usamos unsigned char para el rango de 0 a 255
    unsigned char* input_image = (unsigned char*)malloc(N * M * sizeof(unsigned char));
    // TODO: Cargar la imagen de entrada

    // Reservar memoria en el dispositivo CUDA
    unsigned char* d_input_image, * d_output_image;
    cudaMalloc((void**)&d_input_image, N * M * sizeof(unsigned char));
    cudaMalloc((void**)&d_output_image, N * M * sizeof(unsigned char));

    // Copiar la imagen de entrada desde el host al dispositivo
    cudaMemcpy(d_input_image, input_image, N * M * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Definir los parámetros del kernel
    dim3 block_size(32, 32);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, (M + block_size.y - 1) / block_size.y);

    // Ejecutar el kernel
    threshold_kernel << <grid_size, block_size >> > (d_input_image, d_output_image, N, M, 154);

    // Copiar la imagen resultante desde el dispositivo al host
    unsigned char* output_image = (unsigned char*)malloc(N * M * sizeof(unsigned char));
    cudaMemcpy(output_image, d_output_image, N * M * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Liberar la memoria reservada en el dispositivo
    cudaFree(d_input_image);
    cudaFree(d_output_image);

    // TODO: Guardar la imagen resultante

    return 0;
}

