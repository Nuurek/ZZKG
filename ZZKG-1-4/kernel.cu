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

int factorial(unsigned int n) {
	unsigned int result = 1;

	for (size_t i = 2; i <= n; i++) {
		result *= i;
	}

	return result;
}

void initializeRoute(unsigned int *route, size_t n) {
	for (size_t i = 0; i < n; i++) {
		route[i] = i;
	}
}

const size_t N = 5;
const size_t numberOfPermutations = factorial(N);
unsigned int initialRoute[N];

__global__ void salesman(unsigned int *initialRoute, unsigned int n, unsigned int* output)
{
	const size_t threadId = threadIdx.x;
	const size_t outputIndex = threadId * n;
	unsigned int *factoradic = new unsigned int[n];

	int x = threadId;
	for (size_t i = 1; x != 0; i++) {
		factoradic[n - i] = x % i;
		x = x / i;
	}

	unsigned int * route = new unsigned int[n];
	for (size_t i = 0; i < n; i++) {
		//output[outputIndex + i] = route[factoradic[i]];

		//for (size_t j = factoradic[i]; j < n - 1; j++) {

		//}
		output[outputIndex + i] = factoradic[i];
	}

	delete[] factoradic;
}

int main()
{
	//thrust::host_vector<int> hostInput(arraySize);

	//for (size_t i = 0; i < arraySize; i++) {
	//	hostInput[i] = i + 1;
	//	std::cout << hostInput[i] << ", ";
	//}
	//std::cout << "\n";

	//thrust::device_vector<int> deviceInput = hostInput;
	const size_t arraySize = numberOfPermutations * N;
	initializeRoute(initialRoute, N);
	thrust::device_vector<unsigned int> deviceOutput(arraySize);

	dim3 dimBlock(numberOfPermutations);
	dim3 dimGrid(1);

	salesman << <dimGrid, dimBlock>> > (initialRoute, N, deviceOutput.data().get());

	thrust::host_vector<unsigned int> hostOutput = deviceOutput;

	for (size_t threadId = 0; threadId < numberOfPermutations; threadId++) {
		const size_t outputStart = threadId * N;
		for (size_t i = 0; i < N; i++) {
			std::cout << hostOutput[outputStart + i];
		}
		std::cout << "\n";
	}

    return 0;
}

>>>>>>> 3.3
