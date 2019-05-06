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
#include <thrust/iterator/counting_iterator.h>
#include <random>
#include <iostream>

const size_t vectorSize = 16;

struct isGreaterThanTen : public thrust::binary_function<int, int, int>
{
	__device__ int operator()(int x) {
		return x > 10;
	}
};

int main() {
	thrust::host_vector<int> hostInputNumbers(vectorSize);

	for (size_t i = 0; i < vectorSize; i++) {
		hostInputNumbers[i] = std::rand() % 32;
		std::cout << hostInputNumbers[i]  << ", ";
	}
	std::cout << "\n";

	thrust::counting_iterator<int> deviceFirstIndex = thrust::make_counting_iterator(0);
	thrust::counting_iterator<int> deviceLastIndex = thrust::make_counting_iterator((int)vectorSize);
	thrust::device_vector<int> deviceInputNumbers = hostInputNumbers;
	thrust::device_vector<int> deviceOutput(vectorSize);
	thrust::copy_if(
		deviceFirstIndex,
		deviceLastIndex,
		deviceInputNumbers.begin(), 
		deviceOutput.begin(), 
		isGreaterThanTen()
	);

	thrust::host_vector<int> hostOutput = deviceOutput;

	for (const auto& element : hostOutput) {
		std::cout << element << ", ";
	}
	std::cout << "\n";

	return 0;
}