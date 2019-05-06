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

const size_t vectorSize = 64;
const size_t subvectorSize = 1;
const int maxElement = 10;

__global__ void isSubvectorStartingOnIndex(int* vector, int vectorLength, int* subvector, int subvectorLength, bool* output) {
	const size_t index = threadIdx.x;

	bool result = true;
	for (size_t i = index; i < index + subvectorLength; i++) {
		if (i >= vectorLength || vector[i] != subvector[i]) {
			result = false;
			break;
		}
	}

	output[index] = result;
}

int main() {
	thrust::host_vector<int> hostVector(vectorSize);
	for (auto& element : hostVector) {
		element = std::rand() % maxElement;
		std::cout << element;
	}
	std::cout << "\n";

	thrust::host_vector<int> hostSubvector(subvectorSize);
	for (auto& element : hostSubvector) {
//		element = std::rand() % maxElement;
		element = 2;
		std::cout << element;
	}
	std::cout << "\n";

	thrust::device_vector<int> deviceVector = hostVector;
	thrust::device_vector<int> deviceSubvector = hostSubvector;

	thrust::device_vector<bool> deviceResult(vectorSize);

	dim3 dimBlock(vectorSize);
	dim3 dimGrid(1);

	isSubvectorStartingOnIndex<<<dimGrid, dimBlock>>>(
		deviceVector.data().get(),
		vectorSize,
		deviceSubvector.data().get(),
		subvectorSize,
		deviceResult.data().get()
	);

	for (const auto& element : deviceResult) {
		std::cout << element << ", ";
	}
	std::cout << "\n";
/*
	thrust::counting_iterator<int> deviceFirstIndex = thrust::make_counting_iterator(0);
	thrust::counting_iterator<int> deviceLastIndex = thrust::make_counting_iterator((int)vectorSize);
	thrust::device_vector<int> deviceInputNumbers = hostVector;
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
*/
	return 0;
}