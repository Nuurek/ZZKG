#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <iostream>
#include "kernel.h"

const int N = 32;
const int M = 64;
const int blockSize = 16;

__global__ void add_matrix(float *a, int N, int M)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int index = i + j * N;

	if (i < N && j < M) {
		a[index] += j;
	}
}

int main()
{
	float *a = new float[N*M];

	for (int i = 0; i < N*M; ++i) {		a[i] = 0.0f; 	}
	float *ad;
	const int size = N * M * sizeof(float);
	cudaMalloc((void**)&ad, size);
	
	cudaMemcpy(ad, a, size, cudaMemcpyHostToDevice);
	dim3 dimBlock(blockSize, blockSize);
	const int gridN = N / dimBlock.x + ((N % blockSize) != 0);
	const int gridM = M / dimBlock.y + ((M % blockSize) != 0);
	dim3 dimGrid(gridN, gridM);
	add_matrix <<<dimGrid, dimBlock >> > (ad, N, M);
	cudaMemcpy(a, ad, size, cudaMemcpyDeviceToHost);
	cudaFree(ad);

	for (int j = 0; j < M; ++j) {
		for (int i = 0; i < N; ++i) {
			const int index = j * N + i;
			std::cout << a[index];
		}
		std::cout << "\n";
	}
	std::cout << "\n";

	delete[] a;

    return 0;
}

