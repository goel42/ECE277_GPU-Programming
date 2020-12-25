#include <stdio.h>
#define ROW 80
#define COL 80
__global__ void transpose (float* in, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny) {
        // int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = (threadIdx.y * blockDim.x) + threadIdx.x;
        unsigned int warpIdx = threadId / warpSize;
        if (blockIdx.z == 0 && (blockIdx.x == 2) && (blockIdx.y == 1) && (warpIdx == 0)) {
            in[ix*ny+iy] = 1;
        }
    }
}

int main () {
    float mat[ROW*COL] = {0,};
    float* d_mat = NULL;
    cudaMalloc((void **)&d_mat, ROW*COL * sizeof(float));
    cudaMemcpy(d_mat, mat, ROW*COL*sizeof(float), cudaMemcpyHostToDevice);
    dim3 A(100, 100, 100);
    dim3 B(8, 32);
    transpose<<<A, B>>> (d_mat, ROW, COL);

    cudaMemcpy(mat, d_mat, ROW*COL*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_mat);
    for (int ii = 0; ii < ROW; ii++) {
        for (int jj =0; jj < COL; jj ++)
            printf("%d ", (int) mat[ii*COL+jj]);
        printf("\n");
    }
    return 0;
}