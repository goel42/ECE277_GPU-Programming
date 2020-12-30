/*************************************************************************
/* ECE 277: GPU Programmming 2020 FALL
/* Author and Instructer: Cheolhong An
/* Copyright 2020
/* University of California, San Diego
/*************************************************************************/

#include <stdio.h>
//#define dimension 4
#define num_agent 1

__device__ short *d_action;

__global__ void cuda_agent(int2* cstate, short *d_action) {
	
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	
	short action = 0;

	if (cstate[idx].x == 0 && cstate[idx].y == 0) {
		d_action[idx] = action;
	} // (1,0)
	if (cstate[idx].x == 1 && cstate[idx].y == 0) {
		d_action[idx] = action;
	} // (2,0)
	if (cstate[idx].x == 2 && cstate[idx].y == 0) {
		d_action[idx] = action;
	} // (3,0)
	if (cstate[idx].x == 3 && cstate[idx].y == 0) {
		d_action[idx] = action + 1;
	} // (3,1)
	if (cstate[idx].x == 3 && cstate[idx].y == 1) {
		d_action[idx] = action + 1;
	} // (3,2)
	if (cstate[idx].x == 3 && cstate[idx].y == 2) {
		d_action[idx] = action + 2;
	}
	
}

void agent_init()
{
	// add your codes
	int size = num_agent * sizeof(int);

	cudaMalloc(((void**)&d_action), size);
}

short* agent_action(int2* cstate)
{
	// add your codes
	dim3 block(1, 1, 1);
	dim3 grid(1, 1, 1);

	cuda_agent << <grid, block >> > (cstate, d_action);
	return d_action;
}