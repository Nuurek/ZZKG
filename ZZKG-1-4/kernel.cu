#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cstdio>
#include <iostream>
#include <random>
#include <cmath>
#include "kernel.h"

const int N = 512;
const int blockSize = 16;

struct Input {
	float a;
	float b;
	float c;

	void print() {
		std::cout << "a: " << a << ", b: " << b << ", c: " << c << "\n";
	}
};

struct Output {
	size_t numberOfSolutions;
	float firstSolution;
	float secondSolution;
	float extremum;

	void print() {
		std::cout << "number of solutions: " << numberOfSolutions;
		if (numberOfSolutions > 0) {
			std::cout << ", first solution: " << firstSolution;
		}
		if (numberOfSolutions > 1) {
			std::cout << ", second solution: " << secondSolution;
		}
		std::cout << ", extremum: " << extremum << "\n";
	}
};

__global__ void calculate(Input* inputData, Output* outputData)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	Input input = inputData[index];

	const float delta = input.b * input.b - 4 * input.a * input.c;

	Output output;
	if (delta < 0) {
		output.numberOfSolutions = 0;
	} else if (delta == 0) {
		output.numberOfSolutions = 1;
		output.firstSolution = -input.b / (2 * input.a);
	} else {
		output.numberOfSolutions = 2;
		const float deltaSquared = std::sqrt(delta);
		output.firstSolution = (-input.b - deltaSquared) / (2 * input.a);
		output.secondSolution = (-input.b + deltaSquared) / (2 * input.a);
	}
	output.extremum = -delta / (4 * input.a);

	outputData[index] = output;
}

int main()
{
	thrust::host_vector<Input> hostInput(N);

	for (auto& input : hostInput) {
		input.a = std::rand() % 10 + 1;
		input.b = std::rand() % 10 + 1;
		input.c = std::rand() % 10 + 1;
	}

	thrust::device_vector<Input> deviceInput = hostInput;
	thrust::device_vector<Output> deviceOutput(N);

	const int numberOfThreads = N;
	dim3 dimBlock(blockSize);
	const int gridSize = numberOfThreads / dimBlock.x + ((numberOfThreads % blockSize) != 0);
	dim3 dimGrid(gridSize);
	calculate<<<dimGrid, dimBlock >>> (deviceInput.data().get(), deviceOutput.data().get());

	thrust::host_vector<Output> hostOutput = deviceOutput;

	for (size_t i = 0; i < hostInput.size(); i++) {
		hostInput[i].print();
		hostOutput[i].print();
	}

    return 0;
}

