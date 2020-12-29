/*************************************************************************
/* ECE 277: GPU Programmming 2020 FALL quarter
/* Author and Instructer: Cheolhong An
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
//////////////////////////////////////////////////////////////////////////
#define block_size 256
#define board_size 32 * 32
#define agents 128
//////////////////////////////////////////////////////////////////////////
__device__ float epsilon;
__device__ float *d_epsilon;
__device__ short *d_action;

__device__ curandState *randState;
__device__ float *d_qtable;
__device__ bool *d_alive;

__device__ float gamma = 0.9;
__device__ float alpha = 0.1;
/////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernelInit(curandState *state);
__global__ void agentInit(short *d_agentsActions);
__global__ void qtableInit(float *d_qtable);
__global__ void aliveInit(bool *d_agentAlive);
__global__ void agentAction(int2* cstate, short *d_action, float *d_qtable, curandState *state, float *ep);
__global__ void qtableUpdate(int2* cstate, int2* nstate, float *rewards, bool *d_alive, short *d_action, float *d_qtable, float gamma, float alpha);
__global__ void epsilonInit(float *d_epsilon);
__global__ void updateEps(float *d_epsilon);
/////////////////////////////////////////////////////////////////////////////////////////

void agent_init() // clear action + init Q table + self initialization
{
	// add your codes
	int used_blocksize = agents > block_size ? block_size : agents;
	dim3 block(used_blocksize);
	dim3 grid((agents + block.x - 1) / block.x);
	int action_size = agents * sizeof(short);
	checkCudaErrors(cudaMalloc((void **)&d_action, action_size));
	checkCudaErrors(cudaMalloc((void **)&randState, agents * sizeof(curandState)));
	agentInit << <grid, block >> > (d_action);
	kernelInit << <grid, block >> >(randState);


	int q_table_size = 32 * 32 * 4 * sizeof(float);
	dim3 qBlock(agents / 2, 2);
	dim3 qGrid((32 * 4 + qBlock.x - 1) / qBlock.x, (32 + qBlock.y - 1) / qBlock.y);
	checkCudaErrors(cudaMalloc((void **)&d_qtable, q_table_size));
	qtableInit << <qGrid, qBlock >> > (d_qtable);

	// float eps = 1.0;
	// float alpha = 0.1;
	// float gamma = 0.9;
	checkCudaErrors(cudaMalloc((void **)&d_epsilon, sizeof(float)));
	epsilonInit << <grid, block >> > (d_epsilon);

	int alive_size = agents * sizeof(bool);
	checkCudaErrors(cudaMalloc((void **)&d_alive, alive_size));
	aliveInit << <grid, block >> > (d_alive);
}

void agent_init_episode() // set all agents in active status
{
	// add your codes
	int used_blocksize = agents > block_size ? block_size : agents;
	dim3 block(used_blocksize);
	dim3 grid((agents + block.x - 1) / block.x);
	int alive_size = agents * sizeof(bool);

	checkCudaErrors(cudaMalloc((void **)&d_alive, alive_size));
	aliveInit << <grid, block >> > (d_alive);
}

float agent_adjustepsilon()  // adjust epsilon
{
	// add your codes
	int used_blocksize = agents > block_size ? block_size : agents;
	dim3 block(used_blocksize);
	dim3 grid((agents + block.x - 1) / block.x);
	checkCudaErrors(cudaMemcpy(&epsilon, d_epsilon, sizeof(float), cudaMemcpyDeviceToHost));
	updateEps << <grid, block >> >(d_epsilon);
	return epsilon;
}

short* agent_action(int2* cstate)
{
	// add your codes
	int used_blocksize = agents > block_size ? block_size : agents;
	dim3 block(used_blocksize);
	dim3 grid((agents + block.x - 1) / block.x);
	agentAction << <grid, block >> > (cstate, d_action, d_qtable, randState, d_epsilon);
	return d_action;
}

void agent_update(int2* cstate, int2* nstate, float *rewards)
{
	// add your codes
	int used_blocksize = agents > block_size ? block_size : agents;
	dim3 block(used_blocksize);
	dim3 grid((agents + block.x - 1) / block.x);
	qtableUpdate << <grid, block >> > (cstate, nstate, rewards, d_alive, d_action, d_qtable, gamma, alpha);
}

////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernelInit(curandState *state) {
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	curand_init((unsigned long long)(clock() + idx), idx, 0, &state[idx]);
}
__global__ void agentInit(short *d_agentsActions) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	d_agentsActions[idx] = 0;
}
__global__ void qtableInit(float *d_qtable) {
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int idx = iy * 32 * 4 + ix;
	d_qtable[idx] = 0;
}
__global__ void aliveInit(bool *d_alive) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	d_alive[idx] = true;
}
__global__ void agentAction(int2* cstate, short *d_action, float *d_qtable, curandState *state, float *eps) {
	float d_epsilon = *eps;
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	int cx = cstate[idx].x;
	int cy = cstate[idx].y;
	short action = 0;
	// gama greedy strategy:
	curandState localState = state[idx];
	float base = curand_uniform(&localState);

	if (base < d_epsilon) {
		action = (short)(curand_uniform(&localState) * 4);
	}
	else {
		short action = 0;
		int start_idx = cy * (32 * 32 * 4) + cx * 4;
		for (short i = 0; i <= 3; ++i) {
			int action_idx = start_idx + action;
			int cur_idx = start_idx + i;
			float action_qval = d_qtable[action_idx];
			float cur_qval = d_qtable[cur_idx];
			action = cur_qval > action_qval ? i : action;
		}
	}
	d_action[idx] = action;
	state[idx] = localState;
}
__global__ void qtableUpdate(int2* cstate, int2* nstate, float *rewards, bool *d_alive, short *d_action, float *d_qtable, float gamma, float alpha) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	short cur_action = d_action[idx];
	int cx = cstate[idx].x;
	int cy = cstate[idx].y;

	int nx = nstate[idx].x;
	int ny = nstate[idx].y;

	if (d_alive[idx]) {
		float reward = rewards[idx];
		d_alive[idx] = (reward == 0);
		bool alive = (reward == 0);
		float max_val = 0;

		if (alive) {
			short max_action = 0;
			int start_idx = ny * (32 * 32 * 4) + nx * 4;
			for (short i = 0; i <= 3; ++i) {
				int action_idx = start_idx + max_action;
				int cur_idx = start_idx + i;
				float action_qval = d_qtable[action_idx];
				float cur_qval = d_qtable[cur_idx];
				max_action = cur_qval > action_qval ? i : max_action;
			}
			int max_idx = ny * (32 * 32 * 4) + nx * 4 + max_action;
			max_val = d_qtable[max_idx];
		}

		int cur_idx = cy * (32 * 32 * 4) + cx * 4 + cur_action;
		float delta = gamma * (reward + alpha * max_val - d_qtable[cur_idx]);
		d_qtable[cur_idx] = d_qtable[cur_idx] + delta;
	}
}
__global__ void updateEps(float *d_epsilon) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	d_epsilon[idx] -= 0.001f;

	if (d_epsilon[idx] <= 0.1) {
		d_epsilon[idx] = 0.1;
	}
	return;
}
__global__ void epsilonInit(float *d_epsilon) {
	*d_epsilon = 1.0f;
}