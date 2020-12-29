/*************************************************************************
/* ECE 277: GPU Programmming 2020 FALL
/* Author and Instructer: Cheolhong An
/* Copyright 2020
/* University of California, San Diego
/*************************************************************************/


#include <stdio.h>
#define num_agent 1
// enum ACTION {
// 	right = 0, down, left, up
// 	};

//ACTION Action; 

__device__ short *d_action;


__global__ void cuda_agent(int2* cstate, short *d_action){
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	
	short action = 0;

	if (cstate[idx].x == 0 && cstate[idx].y == 0) {
		d_action[idx] = action;
	} // (1,0)
	if (cstate[idx].x == 1 && cstate[idx].y == 0){
		d_action[idx] = action;
	} // (2,0)
	if (cstate[idx].x == 2 && cstate[idx].y == 0){
		d_action[idx] = action;
	} // (3,0)
	if (cstate[idx].x == 3 && cstate[idx].y == 0){
		d_action[idx] = action + 1;
	}
	if (cstate[idx].x == 3 && cstate[idx].y == 1){
		d_action[idx] = action + 1;
	}
	if (cstate[idx].x == 3 && cstate[idx].y == 2){
		d_action[idx] = action + 2;
	}
	// // // easy way
	// // //short path[16]={0,0,0,1,1,2};
	// // if 0,0,;
	// // cstate x 
	// // if (cstate = sth){
	// // 	action = sth;
	// // } else if(cstate=sth){
	// // 	action = sth; # stop point

	// // }

	// d_action[idx] = action;
}
	

void agent_init()
{
	// add your codes


	int size = num_agent *sizeof(int);

	// allocate a short-type global memory for d_action
	checkCudaErrors(cudaMalloc((void**) &d_action), size);

}

short* agent_action(int2* cstate)
{
	// add your codes
	dim3 block(1,1,1);
	dim3 grid(1,1,1);

	// invokes an CUDA kernel (e.g. cuda agent) for an agent action
	cuda_agent <<<grid, block>>> (cstate, d_action);

	return d_action;
}