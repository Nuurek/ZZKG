#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cstdio>
#include <iostream>
#include <random>
#include <cmath>
#include "kernel.h"

const size_t arraySize = 72363;
const size_t blockSize = 16;

__global__ void calculate(int k, int *data)
{
	int index = k * (blockIdx.x * blockDim.x + threadIdx.x);

	data[index] = data[index] + data[index + k];
}

int main()
{
	thrust::host_vector<int> hostInput(arraySize);

	for (size_t i = 0; i < arraySize; i++) {
		hostInput[i] = i;
		std::cout << hostInput[i] << ", ";
	}
	std::cout << "\n";

	thrust::device_vector<int> deviceInput = hostInput;
	thrust::device_vector<int> deviceOutput(arraySize);

	for (int k = 1; k < arraySize; k *= 2) {
		const size_t numberOfThreads = arraySize / k;
		dim3 dimBlock(blockSize);
		const int gridSize = numberOfThreads / dimBlock.x + ((numberOfThreads % dimBlock.x) != 0);
		dim3 dimGrid(gridSize);
		calculate << <dimGrid, dimBlock >> > (k, deviceInput.data().get());

		thrust::host_vector<int> hostOutput = deviceInput;

		for (const auto& element : hostOutput) {
			std::cout << element << ", ";
		}
		std::cout << "\n";
	}

    return 0;
}

