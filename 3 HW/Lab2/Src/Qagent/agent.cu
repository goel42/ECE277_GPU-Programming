/*************************************************************************
/* ECE 277: GPU Programmming 2020 FALL
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

///////////////////////////////////////////
__device__ float epsilon;
__device__ float d_epsilon;
__device__ short *d_action;
__device__ float *Qtable;
__device__ curandState *randState;

__device__ float DiscountFactor; // gamma, prediction to Q
__device__ float LearningRate;	 // alpha, gradient ascent

///////////////////////////////////////////
//agent_init// create and initialize Qtable, agent self initialization
__global__ void kernelInit(curandState *states);
__global__ void agentInit(short *d_agentActions, int size);
__global__ void QtableInit(float *Qtable, int size);
//agent_adjustepsilon// epsilon adjustment
__global__ void Epsilon_Adjust();
//agent_action// agent action with Qtable
__global__ void AgentAction(int2* cstate, short *d_action, float *Qtable, curandState *states);
//agent_update// agent actio update with rewards
__global__ void Update_Agent(int2* cstate, int2* nstate, float *rewards, short *d_action, float *Qtable);

///////////////////CPU///////////////////////////

void agent_init()
{
	// add your codes
	// create and initialize Q table
	// agent self initialization

	dim3 block(1, 1, 1);
	dim3 grid(1, 1, 1);

	int actionSize = sizeof(int);

	checkCudaErrors(cudaMalloc((void **)&d_action, actionSize));
	checkCudaErrors(cudaMalloc((void **)&randState, sizeof(curandState))); // (*threads_per_block * blocks_per_threads = 1 * 1 = 1)

	agentInit << <grid, block >> > (d_action, 1); // one agent
	kernelInit << <grid, block >> > (randState);
	
	int QtableSize = 4 * 4 * 4 * sizeof(float); // 4x4 dimensions, 4 actions(right, down, left, up)
	checkCudaErrors(cudaMalloc((void **)&Qtable, QtableSize));
	QtableInit << <grid, block >> > (Qtable, 4 * 4 * 4);
		
	// final_epsilon(epsilon) = epsilon - delta_epsilon, 
	// set discount_factor and learning_rate --> to simplify, use cudamemcpytosymbol, not cudamemcpy. 
	float d_ep = 1.0; // delta_epsilon, max. = 1.0
	float df = 0.9; // discount factor, gamma = 0.9
	float lr = 0.1; // learning rate, alpha = 0.1
	checkCudaErrors(cudaMemcpyToSymbol(d_epsilon, &d_ep, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(DiscountFactor, &df, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(LearningRate, &lr, sizeof(float)));
}

float agent_adjustepsilon()
{
	// adjust epsilon
	// add your codes
	dim3 block(1, 1, 1);
	dim3 grid(1, 1, 1);
	Epsilon_Adjust << <grid, block >> > ();
	float epsilon = 0.0f;
	checkCudaErrors(cudaMemcpyFromSymbol(&epsilon, d_epsilon, sizeof(float)));

	return epsilon;
}

short* agent_action(int2* cstate)
{
	// agent action
	// add your codes
	dim3 block(1, 1, 1);
	dim3 grid(1, 1, 1);

	AgentAction << <grid, block >> > (cstate, d_action, Qtable, randState);

	return d_action;
}

void agent_update(int2* cstate, int2* nstate, float *rewards)
{
	// agent update with cs, ns, rewards <- d_action, Qtable (Qtable)
	// add your codes
	dim3 block(1, 1, 1);
	dim3 grid(1, 1, 1);
	Update_Agent << <grid, block >> > (cstate, nstate, rewards, d_action, Qtable);
}

///////////////////GPU/////////////////////

__global__ void kernelInit(curandState *states) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init((long)(clock() + idx), idx, 0, &states[idx]); //reference: lecture materials 
}

__global__ void agentInit(short *d_agentActions, int size) {
	for (int i = 0; i < size; ++i) {
		d_agentActions[i] = 0;
	}
}

__global__ void QtableInit(float *Qtable, int size) {
	for (int i = 0; i < size; ++i) {
		Qtable[i] = 0;
	}
}

__global__ void AgentAction(int2* cstate, short *d_action, float *Qtable, curandState *states) {
	int idx = threadIdx.x + blockDim.x*blockIdx.x;

	int cx = cstate[idx].x;
	int cy = cstate[idx].y;
	short action = 0;

	// local position of agent
	curandState local = *states;
	float base = curand_uniform(&local);

	// agent moving
	if (base < d_epsilon) {
		float action_base = curand_uniform(&local) * 4; // number of actions
		action = (short)action_base;
	}
	else {
		for (short i = 0; i <= 3; ++i) {
			int table_Idx = i * (4 * 4) + cy * 4 + cx; // table index
			int action_Idx = action * (4 * 4) + cy * 4 + cx; // action index (action * (dimension^2) + current y position * 4 + current x position)
			action = Qtable[table_Idx] > Qtable[action_Idx] ? i : action;
		}
	}
	d_action[idx] = action;
	*states = local;
}

__global__ void Update_Agent(int2* cstate, int2* nstate, float *rewards, short *d_action, float *Qtable) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	int cx = cstate[idx].x;
	int cy = cstate[idx].y;

	int nx = nstate[idx].x;
	int ny = nstate[idx].y;

	// rewards is 1, -1, or 0 case;
	// rewards is 1 (flag) or -1 (mines) --> return;
	// update Qval and Qtable
	if (cx == nx && cy == ny && rewards[idx] != 0) {
		return;
	}
	else {
		float Q_max = 0.0f;
		for (short i = 0; i <= 3; ++i) {
			int table_Idx = i * (4 * 4) + ny * 4 + nx;
			float Qtable_Idx_val = Qtable[table_Idx]; // Qtable creation and update (below)
			Q_max = Qtable_Idx_val > Q_max ? Qtable_Idx_val : Q_max;
		}

		int table_Idx = d_action[idx] * (4 * 4) + cy * 4 + cx;
		float current_Q_val = Qtable[table_Idx];
		Qtable[table_Idx] = current_Q_val + LearningRate * (rewards[idx] + DiscountFactor * Q_max - current_Q_val);
	}
}


__global__ void Epsilon_Adjust() {
	d_epsilon -= 0.001f;

	if (d_epsilon <= 0.1) {
		d_epsilon = 0.1;
	}
}