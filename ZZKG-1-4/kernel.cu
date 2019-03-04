#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cstdio>
#include <iostream>
#include <random>
#include <cmath>
#include "kernel.h"

const size_t arraySize = 5000;
const size_t blockSize = 16;
const size_t shift = 10;

__global__ void calculate(float *input, float *output)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < arraySize) {
		output[(index + shift) % arraySize] = input[index];
	}
}

int main()
{
	thrust::host_vector<float> hostInput(arraySize);

	for (size_t i = 0; i < arraySize; i++) {
		hostInput[i] = i;
		std::cout << hostInput[i] << ", ";
	}
	std::cout << "\n";

	thrust::device_vector<float> deviceInput = hostInput;
	thrust::device_vector<float> deviceOutput(arraySize);

	dim3 dimBlock(blockSize);
	const int gridSize = arraySize / dimBlock.x + ((arraySize % dimBlock.x) != 0);
	dim3 dimGrid(gridSize);
	calculate<<<dimGrid, dimBlock >>> (deviceInput.data().get(), deviceOutput.data().get());

	thrust::host_vector<float> hostOutput = deviceOutput;

	for (const auto& element : hostOutput) {
		std::cout << element << ", ";
	}
	std::cout << "\n";

    return 0;
}

