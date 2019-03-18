<<<<<<< HEAD
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

=======
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cstdio>
#include <iostream>
#include <random>
#include <cmath>
#include "kernel.h"

const size_t arraySize = 64;
const size_t blockSize = 64;

__global__ void inclusiveScan(int *input, int* output, int log)
{
	__shared__ int shared[arraySize];

	const size_t threadId = threadIdx.x;
	const size_t partSize = arraySize / blockSize;
	for (size_t i = threadId * partSize; i < (threadId + 1) * partSize; i++) {
		shared[i] = input[i];
	}
	
	__syncthreads();

	int previousSum = 0;

	for (size_t i = 0; i < threadId * partSize; i++) {
		previousSum += shared[i];
	}

	__syncthreads();

	for (size_t i = 0; i <= log; i++) {
		const int power = 1 << i;
		if (threadId >= power) {
			shared[threadId] += shared[threadId - power];
		}

		__syncthreads();
	}

	__syncthreads();

	for (size_t i = threadId * partSize; i < (threadId + 1) * partSize; i++) {
		output[i] = shared[i];
	}
}

int log2(int n) {
	int exponent = 1;
	int power = 0;

	while (exponent <= n) {
		exponent = 1 << power++;
	}

	return power;
}

int main()
{
	thrust::host_vector<int> hostInput(arraySize);

	for (size_t i = 0; i < arraySize; i++) {
		hostInput[i] = i + 1;
		std::cout << hostInput[i] << ", ";
	}
	std::cout << "\n";

	thrust::device_vector<int> deviceInput = hostInput;
	thrust::device_vector<int> deviceOutput(arraySize);

	dim3 dimBlock(blockSize);
	dim3 dimGrid(1);
	const int log = log2(arraySize - 1);

	std::cout << "log2(" << arraySize << ") = " << log << "\n";

	inclusiveScan << <dimGrid, dimBlock>> > (deviceInput.data().get(), deviceOutput.data().get(), log);

	thrust::host_vector<int> hostOutput = deviceOutput;

	for (const auto& element : hostOutput) {
		std::cout << element << ", ";
	}
	std::cout << "\n";

    return 0;
}

>>>>>>> 3.3
