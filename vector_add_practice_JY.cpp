#include <stdio.h>

__global__ void cudaadd(int* vec3, int* vec2, int* vec1, int N) {
    //for (int ii = 0; ii < 10; ii ++) {
        int ii = blockIdx.x *blockDim.x + threadIdx.x;
        if(ii >= N)
            return;

        vec3[ii] = vec1[ii]  + vec2[ii] ;
    //}
}


int main() {
    int * vec1 = (int * ) malloc(sizeof(int) * 10);
    int * vec2 = (int * ) malloc(sizeof(int) * 10);
    for (int ii = 0; ii < 10; ii ++) {
        vec1[ii] = ii;
        vec2[ii] = ii * ii ;
    }

    int * h_vec3 = (int * ) malloc(sizeof(int) * 10);
    int * vec3 = (int * ) malloc(sizeof(int) * 10);
    for (int ii = 0; ii < 10; ii ++) {
        vec3[ii] = vec1[ii]  + vec2[ii] ;
    }
    for (int ii = 0; ii < 10; ii ++) {
        printf("%d ", vec3[ii]);
    }
    printf("\n");

    int* d_vec1 = NULL;
    cudaMalloc( (void**)&d_vec1, 10 * sizeof(int) );
    int* d_vec2 = NULL;
    cudaMalloc( (void**)&d_vec2, 10 * sizeof(int) );
    int* d_vec3 = NULL;
    cudaMalloc( (void**)&d_vec3, 10 * sizeof(int) );
    cudaMemcpy(d_vec1, vec1, 10 * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy(d_vec2, vec2, 10 * sizeof(int), cudaMemcpyHostToDevice );

    cudaadd<<<1, 1024>>>(d_vec3, d_vec2, d_vec1, 10);
    cudaMemcpy(h_vec3, d_vec3, 10 * sizeof(int), cudaMemcpyDeviceToHost );
     for (int ii = 0; ii < 10; ii ++) {
        printf("%d ", h_vec3[ii]);
    }
    printf("\n");
    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_vec3);
    free(vec1);
    free(vec2);
    free(vec3);
    free(h_vec3);

    return 0;
}