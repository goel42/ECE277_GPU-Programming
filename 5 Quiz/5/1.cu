#include <stdio.h>

/*
 * Example kernels for transposing a rectangular host array using a variety of
 * optimizations, including shared memory, unrolling, and memory padding.
 */

#define BDIMX 16
#define BDIMY 8

void initialData(bool *in, const int size)
{
	for (int i = 0; i < size; i++) {
		in[i] = 0;
	}

	return;
}

__global__ void transposeSmem(bool *out, bool *in, int nx, int ny)
{
	// static shared memory
	__shared__ bool tile[BDIMY][BDIMX];

	// coordinate in original matrix
	unsigned int ix, iy, ti, to;
	ix = blockDim.x * blockIdx.x + threadIdx.x;
	iy = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int threadId = (threadIdx.y * blockDim.x) + threadIdx.x;
	unsigned int warpIdx = threadId / warpSize;
	// linear global memory index for original matrix
	ti = iy * nx + ix;

	// thread index in transposed block
	unsigned int bidx, irow, icol;
	bidx = threadIdx.y * blockDim.x + threadIdx.x;
	irow = bidx / blockDim.y;
	icol = bidx % blockDim.y;

	// coordinate in transposed matrix
	ix = blockDim.y * blockIdx.y + icol;
	iy = blockDim.x * blockIdx.x + irow;

	// linear global memory index for transposed matrix
	to = iy * ny + ix;

	// transpose with boundary test
	if (ix < nx && iy < ny)
	{
		// load data from global memory to shared memory
		if (blockIdx.z == 0 && (blockIdx.x == 2) && (blockIdx.y == 1) && (warpIdx == 0)) {
			in[ti] = 1;
		}
		tile[threadIdx.y][threadIdx.x] = in[ti];
		

		// thread synchronization
		__syncthreads();

		// store data to global memory from shared memory
		
		out[to] = tile[icol][irow];
		/*if (blockIdx.z == 0 && (blockIdx.x == 2) && (blockIdx.y == 1) && (warpIdx == 0)) {
			out[to] = 1;
		}*/
	}
}


int main(int argc, char **argv)
{
	// set up array size 2048
	int nx = 50;
	int ny = 50;

	size_t nBytes = nx * ny * sizeof(bool);

	bool *d_A, *d_C;
	cudaMalloc((bool**)&d_A, nBytes);
	cudaMalloc((bool**)&d_C, nBytes);

	// execution configuration
	dim3 block(BDIMX, BDIMY);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	dim3 grid2((nx + block.x * 2 - 1) / (block.x * 2),
		(ny + block.y - 1) / block.y);

	// allocate host memory
	bool *h_A = (bool *)malloc(nBytes);
	bool *hostRef = (bool *)malloc(nBytes);
	bool *gpuRef = (bool *)malloc(nBytes);

	//  initialize host array
	initialData(h_A, nx * ny);
	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	// tranpose smem
	cudaMemset(d_C, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	transposeSmem << <grid, block >> > (d_C, d_A, nx, ny);
	cudaDeviceSynchronize();

	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			printf("%d ", gpuRef[i * nx + j]);
		}
		printf("\n");
	}
	printf("\n");


	// free host and device memory
	cudaFree(d_A);
	cudaFree(d_C);
	free(h_A);
	free(hostRef);
	free(gpuRef);

	// reset device
	(cudaDeviceReset());
	return EXIT_SUCCESS;
}