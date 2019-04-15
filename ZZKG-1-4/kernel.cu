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

struct myMax : public thrust::binary_function<int, int, int>
{
	__host__ __device__ int operator()(int x, int y) {
		return x < y ? y : x;
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

	thrust::inclusive_scan(deviceInput.begin(), deviceInput.end(), deviceOutput.begin(), myMax());

	for (const auto& element : deviceOutput) {
		std::cout << element << ", ";
	}
	std::cout << "\n";

	return 0;
}