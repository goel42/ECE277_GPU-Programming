/*
 * EXAMPLE OF PATTERN CONVOLUTION CHAPTER 7
 */

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define CHECK_ERROR(call) { \
cudaError_t err = call; \
if (err != cudaSuccess) { \
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(err); \
    } \
}

// compute vector convoluiton
// each thread performs one pair-wise convolution
__global__
void convolution_1D_basic_kernel(float *d_N, float *d_M, float *d_P, int Mask_Width, int Width){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int N_start_point = i - Mask_Width/2;
    
    float Pvalue = 0;
    
    if (i < Width) {
        for (int j = 0; j < Mask_Width; j++) {
            if (N_start_point + j >= 0 && N_start_point + j < Width) {
                Pvalue += d_N[N_start_point + j] * d_M[j];
            }
        }
        d_P[i] = Pvalue;
    }
}

float convolution_1D_basic(float *h_N, float *h_M, float *h_P, int Mask_Width, int Width) {
    
    float *d_N, *d_M, *d_P;
    int sizeWidth = Width*sizeof(float);
    int sizeMask_Width = Mask_Width*sizeof(float);
    
    cudaEvent_t startTimeCuda, stopTimeCuda;
    cudaEventCreate(&startTimeCuda);
    cudaEventCreate(&stopTimeCuda);
    
    //1. Allocate global memory on the device for N, M and P
    CHECK_ERROR(cudaMalloc((void**)&d_N, sizeWidth));
    CHECK_ERROR(cudaMalloc((void**)&d_P, sizeWidth));
    CHECK_ERROR(cudaMalloc((void**)&d_M, sizeMask_Width));
    
    // copy N and M to device memory
    cudaMemcpy(d_N, h_N, sizeWidth, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, sizeMask_Width, cudaMemcpyHostToDevice);
    
    //2. Kernel launch code - to have the device to perform the actual convolution
    // ------------------- CUDA COMPUTATION ---------------------------
    cudaEventRecord(startTimeCuda, 0);
    
    dim3 dimGrid(ceil((float)Width / (float)Mask_Width),1,1);
    dim3 dimBlock(Mask_Width,1,1);
    convolution_1D_basic_kernel<<<dimGrid, dimBlock>>>(d_N, d_M, d_P, Mask_Width, Width);
    
    cudaEventRecord(stopTimeCuda, 0);
    
    // ---------------------- CUDA ENDING -----------------------------
    cudaEventSynchronize(stopTimeCuda);
    float msTime;
    cudaEventElapsedTime(&msTime, startTimeCuda, stopTimeCuda);
    printf("KernelTime: %f\n", msTime);
    
    //3. copy C from the device memory
    cudaMemcpy(h_P, d_P, sizeWidth, cudaMemcpyDeviceToHost);
    
    // Free device vectors
    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);
    
    return msTime;
}

void printArray(float *A, int size){
    for (int i = 0; i < size; i++) {
        printf("%.2f ", A[i]);
    }
    printf("\n");
}

void sequentialConv(float *h_N, float *h_M, float *h_PS, int n, int Mask_Width){
    float Pvalue;
    for (int i = 0, pos; i < n; i++) {
        Pvalue = 0.0f;
        pos = i - Mask_Width/2;
        for (int j = 0; j < Mask_Width; j++) {
            if (j + pos >= 0 && j + pos < n)
                Pvalue += h_N[j + pos] * h_M[j];
        }
        h_PS[i] = Pvalue;
    }
}

int main(void) {
    
    // create and host vectors
    float *h_P, *h_N, *h_PS;
    const float val = (float)1/(float)2;
    const int n = 100000;
    const int Mask_Width = 3;
    float h_M[] = {-val, 0, val}; // the mask
    float msTime, msTime_seq;
    cudaEvent_t startTimeCuda, stopTimeCuda;
    
    cudaEventCreate(&startTimeCuda);
    cudaEventCreate(&stopTimeCuda);
    
    // allocate memory for host vectors
    //h_M = (int*)malloc(sizeof(int)*Mask_Width);   // mask array
    h_N = (float*)malloc(sizeof(float)*n);          // input array
    h_P = (float*)malloc(sizeof(float)*n);          // output array
    h_PS = (float*)malloc(sizeof(float)*n);         // output array sequential result
    
    // set initial values for vectors
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        h_N[i] = i + 1;
        //h_N[i] = ((float)rand() / (float)(RAND_MAX)) * 100;
    }
    
    // -------------------------- parrallel convolution -----------------------------------
    msTime = convolution_1D_basic(h_N, h_M, h_P, Mask_Width, n);
    
    // -------------------------- perform sequential convolution --------------------------
    cudaEventRecord(startTimeCuda, 0);
    sequentialConv(h_N, h_M, h_PS, n, Mask_Width);
    cudaEventRecord(stopTimeCuda, 0);
    cudaEventSynchronize(stopTimeCuda);
    cudaEventElapsedTime(&msTime_seq, startTimeCuda, stopTimeCuda);
    printf("HostTime: %f\n", msTime_seq);
    
    /*
     printf("----------------- ARRAY INPUT -----------------\n");
     printArray(h_N, n);
     
     printf("---------- ARRAY RESULT - SEQUENTIAL ----------\n");
     printArray(h_PS, n);
     
     printf("---------- ARRAY RESULT - PARALLEL ------------\n");
     printArray(h_P, n);
     */
    
    
    // verify the result
    for (int i = 0; i < n; i++) {
        if(h_P[i] != h_PS[i]){
            printf("\x1b[31mError\x1b[0m into result: h_P[%d] = %.2f != %.2f = h_PS[%d]\n", i, h_P[i], h_PS[i], i);
            goto Error;
        }
    }
    
    printf("Ok convolution completed with \x1b[32msuccess\x1b[0m!\n\n");
    printf("Speedup: %f\n", msTime_seq/msTime);
    
    // Free host memory
    free(h_N);
    free(h_P);
    free(h_PS);
    
    return 0;
    
Error:
    free(h_N);
    free(h_P);
    free(h_PS);
    return -1;
}
