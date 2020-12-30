/*************************************************************************
/* ECE 277: GPU Programmming 2020 FALL quarter
/* Author and Instructer: Cheolhong An
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <stdio.h>
#include "agent.h"
///////////////////////////////////////////////////////////////////////////
#define block_size 128 // Improvement: adjust grid and block size
#define board_size 46
#define num_agent 512
///////////////////////////////////////////////////////////////////////////
__inline__ __device__ 
short find_max_action(float* d_qtable, int x, int y) {
	short action = 0;
	#pragma unroll
	for (short i = 0; i <= 3; ++i) {
		int action_idx = y * (board_size * 4) + x * 4 + action;
		int cur_idx = y * (board_size * 4) + x * 4 + i;
		float action_val = d_qtable[action_idx];
		float cur_val = d_qtable[cur_idx];
		action = cur_val > action_val ? i : action;
	}
	return action; // return the maximum action value
}
///////////////////////////GPU functions///////////////////////////////////
__global__ void kernelInit(curandState *state) {
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	curand_init((unsigned long long)(clock() + idx), idx, 0, &state[idx]);
}

__global__ void agentInit(short *d_agentsActions) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	d_agentsActions[idx] = 0;
}

__global__ void qtableInit(float *d_qtable) {
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	int idx = iy * board_size * 4 + ix;
	d_qtable[idx] = 0;
}

__global__ void aliveInit(bool *d_alive) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	d_alive[idx] = true;
}

__global__ void agentAction(int2* cstate, short *d_action, float *d_qtable, curandState *state, float *eps) {
	float d_epsilon = *eps;
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	int x = cstate[idx].x;
	int y = cstate[idx].y;

	short action = 0;
	curandState local = state[idx];
	float base = curand_uniform(&local);

	if (base < d_epsilon) {
		action = (short) (curand_uniform(&local) * 4);
	} else {
		action = find_max_action(d_qtable, x, y);
	}
	d_action[idx] = action;
	state[idx] = local;
}

__global__ void qtableUpdate(int2* cstate, int2* nstate, float *rewards, bool *d_alive, short *d_action, float *d_qtable, float gamma, float alpha) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	
	short cur_action = d_action[idx];
	int cx = cstate[idx].x;
	int cy = cstate[idx].y;

	int nx = nstate[idx].x;
	int ny = nstate[idx].y;
	
	if (d_alive[idx]) {
		float r = rewards[idx];
		d_alive[idx] = (r == 0);
		bool alive = (r == 0);
		float max_val = 0;
	
		if (alive) {
			short max_action = find_max_action(d_qtable, nx, ny);
			int max_idx = ny * (board_size * 4) + nx * 4 + max_action;
			max_val = d_qtable[max_idx];
		}
		// update formula
		int cur_idx = cy * (board_size * 4) + cx * 4 + cur_action;
		d_qtable[cur_idx] = d_qtable[cur_idx] + alpha * (r + gamma * max_val - d_qtable[cur_idx]);
	}
}

__global__ void updateEps(float *d_epsilon) {
	float val = *d_epsilon; // Improvement: No need to use idx
	
	if (val < 0.001){
		(*d_epsilon) = 0;
	} else {
		(*d_epsilon) = val - 0.005f; // Improvement: adjust epsilon decay rate
	}
}

__global__ void epsilonInit(float *d_epsilon) {
	*d_epsilon = 1.0f;
}
/////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////CPU functions//////////////////////////////////////////
void Agent::init_New() { // agent_init_episode
	int used_blocksize = num_agent > block_size ? block_size : num_agent;
	block = dim3(used_blocksize);
	grid = dim3((num_agent + block.x - 1) / block.x);
	
	this->init_Variables();
	this->init_Agents();
	this->init_Qtable();
}

void Agent::init_Agents() {
	int action_size = num_agent * sizeof(short);
	int alive_size = num_agent * sizeof(bool);

	checkCudaErrors(cudaMalloc((void **)&d_action, action_size));
	checkCudaErrors(cudaMalloc((void **)&d_alive, alive_size));
	checkCudaErrors(cudaMalloc((void **)&randState, num_agent * sizeof(curandState)));

	agentInit <<<grid, block>>> (d_action);
	aliveInit <<<grid, block >>> (d_alive);
	kernelInit <<<grid, block >>>(randState);
}

void Agent::init_Qtable() {
	// init qtable, agent action states
	int qtable_size = board_size * board_size * 4 * sizeof(float);
	checkCudaErrors(cudaMalloc((void **)&d_qtable, qtable_size));
	
	dim3 qBlock(num_agent / 2, 2);
	dim3 qGrid((board_size * 4 + qBlock.x - 1) / qBlock.x, (board_size + qBlock.y - 1) / qBlock.y);
	qtableInit <<<qGrid, qBlock >>> (d_qtable);
}

void Agent::init_Variables() {
	float eps = 1.0;
	checkCudaErrors(cudaMalloc((void **)&d_epsilon, sizeof(float)));
	epsilonInit <<<1, 1 >>> (d_epsilon);

	this->alpha = 0.5f;
	this->gamma = 0.3f;
}

float Agent::update_Epsilon() {
	updateEps <<<1, 1 >>>(this->d_epsilon);
	float epsilon = 0.0f;
	checkCudaErrors(cudaMemcpy(&epsilon, d_epsilon, sizeof(float), cudaMemcpyDeviceToHost));
	return epsilon;
}

void Agent::action_Taken(int2* cstate) {
	agentAction <<<grid, block >>> (cstate, d_action, d_qtable, randState, this->d_epsilon);
}

void Agent::update_Agents(int2* cstate, int2* nstate, float *rewards) {
	qtableUpdate <<<grid, block >>> (cstate, nstate, rewards, d_alive, d_action, d_qtable, gamma, alpha);
}