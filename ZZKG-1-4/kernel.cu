#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <random>
#include <iostream>

const size_t vectorSize = 16;

__global__ void mapToBits(int* input, int length, int* output) {
	const size_t index = threadIdx.x;
	output[index] = __popc(input[index]);
}

struct sumBits : public thrust::binary_function<int, int, int>
{
	__device__ int operator()(int x, int y) {
		return x + y;
	}
};

int main() {
	thrust::host_vector<int> hostInput(vectorSize);

	for (auto& element : hostInput) {
		element = std::rand();
		std::cout << element << ", ";
	}
	std::cout << "\n";

	thrust::device_vector<int> deviceInput = hostInput;
	thrust::device_vector<int> deviceOutput(vectorSize);

	dim3 dimBlock(vectorSize);
	dim3 dimGrid(1);
	mapToBits<<<dimGrid, dimBlock>>>(deviceInput.data().get(), vectorSize, deviceOutput.data().get());

	thrust::host_vector<int> hostBits = deviceOutput;
	for (const auto& bits : hostBits) {
		std::cout << bits << ", ";
	}
	std::cout << "\n";
	thrust::device_vector<int> deviceBits = hostBits;

	int summedBits = thrust::reduce(deviceBits.begin(), deviceBits.end(), 0);
	std::cout << summedBits << "\n";

	return 0;
}