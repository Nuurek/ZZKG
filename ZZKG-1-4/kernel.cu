#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <iostream>

const int N = 1024;
const int blockSize = 16;

__global__ void add_matrix(float *a, float *b, float *c, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int index = i + j * N;
	if (i < N && j < N) c[index] = a[index] + b[index];
}

int main()
{
	float *a = new float[N*N];
	float *b = new float[N*N];
	float *c = new float[N*N];

	for (int i = 0; i < N*N; ++i) { a[i] = 1.0f; b[i] = 3.5f; }
	float *ad, *bd, *cd;
	const int size = N * N * sizeof(float);
	cudaMalloc((void**)&ad, size);
	cudaMalloc((void**)&bd, size);
	cudaMalloc((void**)&cd, size);

	cudaMemcpy(ad, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(bd, b, size, cudaMemcpyHostToDevice);
	dim3 dimBlock(blockSize, blockSize);
	dim3 dimGrid(N / dimBlock.x, N / dimBlock.y);
	add_matrix <<<dimGrid, dimBlock >> > (ad, bd, cd, N);
	cudaMemcpy(c, cd, size, cudaMemcpyDeviceToHost);
	cudaFree(ad); cudaFree(bd); cudaFree(cd);
	
	for (int i = 0; i < N * N; ++i) {
		std::cout << c[i];
	}
	std::cout << "\n";

	delete[] a; delete[] b; delete[] c;

    return 0;
}

