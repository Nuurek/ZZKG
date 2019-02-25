#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <iostream>
#include "kernel.h"

const int N = 512;
const int blockSize = 16;

__global__ void generate_matrix(float *a, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int index = i + j * N;

	if (i < N && j < N) {
		a[index] = i * j;
	}
}

int main()
{
	float *a = new float[N*N];

	for (int i = 0; i < N*N; ++i) {		a[i] = 0.0f; 	}
	float *ad;
	const int size = N * N * sizeof(float);
	cudaMalloc((void**)&ad, size);
	
	cudaMemcpy(ad, a, size, cudaMemcpyHostToDevice);
	dim3 dimBlock(blockSize, blockSize);
	const int gridSize = N / dimBlock.x + ((N % blockSize) != 0);
	dim3 dimGrid(gridSize, gridSize);
	generate_matrix<<<dimGrid, dimBlock >>>(ad, N);
	cudaMemcpy(a, ad, size, cudaMemcpyDeviceToHost);
	cudaFree(ad);

	for (int j = 0; j < N; ++j) {
		for (int i = 0; i < N; ++i) {
			const int index = j * N + i;
			std::cout << a[index] << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";

	delete[] a;

    return 0;
}

