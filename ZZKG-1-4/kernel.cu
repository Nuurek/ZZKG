#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <iostream>
#include "kernel.h"

const int N = 144;
const int blockSize = 16;

__global__ void swapMatrix(float *a, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int index = i + j * N;
	const int numberOfThreads = N / 2;

	if (index < numberOfThreads) {
		const int toSwapIndex = N - 1 - index;
		const float toSwap = a[toSwapIndex];
		a[toSwapIndex] = a[index];
		a[index] = toSwap;
	}
}

int main()
{
	float *a = new float[N];

	for (int i = 0; i < N; ++i) {		a[i] = i;
		std::cout << a[i] << " ";	}
	std::cout << "\n";
	float *ad;
	const int size = N * sizeof(float);
	cudaMalloc((void**)&ad, size);
	
	cudaMemcpy(ad, a, size, cudaMemcpyHostToDevice);
	const int numberOfThreads = N / 2;
	dim3 dimBlock(blockSize);
	const int gridSize = numberOfThreads / dimBlock.x + ((numberOfThreads % blockSize) != 0);
	dim3 dimGrid(gridSize);
	swapMatrix<<<dimGrid, dimBlock >>>(ad, N);
	cudaMemcpy(a, ad, size, cudaMemcpyDeviceToHost);
	cudaFree(ad);

	for (int i = 0; i < N; ++i) {
		std::cout << a[i] << " ";
	}
	std::cout << "\n";

	delete[] a;

    return 0;
}

