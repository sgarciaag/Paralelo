#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <vector>
#include<ctime>

using namespace std;

__global__ void searchAlgorithm(int* array, int* pos, int target)
{
    int id = threadIdx.x;
    //__syncthreads();
    if (array[id] == target) {
        pos[0] = id;
    }

}

int main()
{
    const int size = 100;
    int target = 33;
    //array init
    int* array_host = (int*)malloc(size * sizeof(int));
    int* array_device;
    cudaMalloc(&array_device, size * sizeof(int));

    //aux inot
    vector <int> nums(size);

    //pos init
    int* pos_host = (int*)malloc(sizeof(int));
    int* pos_device;
    cudaMalloc(&pos_device, sizeof(int));
    pos_host[0] = -1;

    srand(time(0));

    for (int i = 0; i < size; i++)
    {
        nums[i] = rand() % 150 + 1;

    }
    //nums[0] = target;

    sort(nums.begin(), nums.end());

    for (int i = 0; i < size; i++)
    {
        array_host[i] = nums[i];
        cout << i << ": " << array_host[i] << endl;
    }

    cudaMemcpy(array_device, array_host, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pos_device, pos_host, sizeof(int), cudaMemcpyHostToDevice);
    searchAlgorithm << <1, size >> > (array_device, pos_device, target);
    cudaMemcpy(array_host, array_device, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(pos_host, pos_device, sizeof(int), cudaMemcpyDeviceToHost);

    if (pos_host[0] == -1) {
        cout << "target number " << target << " wasn't found";
    }
    else {

        cout << target << " is on position " << pos_host[0] << ": " << array_host[pos_host[0]];
    }

    free(pos_host);
    free(array_host);
    cudaFree(pos_device);
    cudaFree(array_device);

    return 0;
}