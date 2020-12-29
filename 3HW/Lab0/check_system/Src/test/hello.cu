#include <stdio.h>

__global__ void mykernel(void) {
		// Print your name, site number
	}

	int main(void) {
		 mykernel <<<1,1>>>();
		printf("Hello World!\n");
		return 0;
	}
