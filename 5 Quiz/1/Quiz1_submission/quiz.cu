#include <stdio.h>

__global__ void vector_add(int *a, int *b, int *c)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x; /* FIXME */
    c[index] = a[index] + b[index];
}


#define N (2048*2048)
#define THREADS_PER_BLOCK 512

int main()
{
  int *a, *b, *c, *golden;
    int *d_a, *d_b, *d_c;

    int size = N * sizeof( int );

    /* allocate space for device copies of a, b, c */
    cudaMalloc( (void **) &d_a, size );
    cudaMalloc( (void **) &d_b, size );
    cudaMalloc( (void **) &d_c, size );

    /* allocate space for host copies of a, b, c and setup input values */

    a = (int *)malloc( size );
    b = (int *)malloc( size );
    c = (int *)malloc( size );
    golden = (int *)malloc(size);

    for( int i = 0; i < N; i++ )
    {
        a[i] = b[i] = i;
        golden[i] = a[i] + b[i];
        c[i] = 0;
    }

    /* copy inputs to device */
    /* fix the parameters needed to copy data to the device */
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice); /* FIXME */
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice); /* FIXME */

    /* launch the kernel on the GPU */
    /* insert the launch parameters to launch the kernel properly using blocks and threads */ 
    vector_add<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( d_a, d_b, d_c ); /* FIXME */

    /* copy result back to host */
    /* fix the parameters needed to copy data back to the host */
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost); /* FIXME */


    bool pass = true;
    for (int i = 0; i < N; i++) {
        if (golden[i] != c[i])
            pass = false;
    }

    if (pass)
        printf("PASS\n");
    else
        printf("FAIL\n");

    printf("Haenara Shin, A53233226, 29\n");

    free(a);
    free(b);
    free(c);
    free(golden);
    cudaFree( d_a );
    cudaFree( d_b );
    cudaFree( d_c );
    
    return 0;
} 
