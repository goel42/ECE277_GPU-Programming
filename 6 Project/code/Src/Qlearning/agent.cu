#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include "common_def.h"
#include "draw_env.h"
#include <stdio.h>
#define uniform() (rand()/(RAND_MAX + 1.0))
#define SCEIL(x,n) ((x + (1 << n) - 1) >> n)
#define XCEIL(x,n) ( SCEIL(x,n) << n)
#define NUM_ACTIONS 4
#define LOG2_AGENT_STRIDE 7 
#define FLAG_TARGET 25
float* d_Qtable;
short* d_action;
float* d_Qtabla_diff;
int2* d_flag_table;


//policy
float* d_agent_epsilon;

float epsilon;
float min_epsilon;
float alpha = 0.1;
float gamma = 0.8;
float learning_rate;

int* d_active_agent;
int* d_flag_agent;
int *d_Alive_agent;
int *d_total_rewards;

curandState_t* rand_states_action;
bool global_start;
int total_rewards[NUM_AGENT];

/////////////////////////////////////
__global__ void q_table_init_cuda_random(curandState_t* states, float* qtable, int board_size);
__global__ void q_table_rand_init(unsigned int seed, curandState_t* states, int board_size);

__global__ void agent_action_cuda_random(int2* cstate, curandState_t* states, short *action, float* qtable
	, int* alive_agent, int* total_rewards, float* agent_epsilon, int board_size);
__global__ void rand_init(unsigned int seed, curandState_t* states);
__global__ void agent_cal_diff_cuda(int2* cstate, int2* nstate, float *rewards, float *qtable, float* qtabld_diff, short *action, int* alive_agent, int* total_rewards,
	int2* flag_table, float* agent_epsilon, float learning_rate);
__global__ void agent_update_qtable_cuda(int2* cstate, int2* nstate, float *rewards, float *qtable, float* qtabld_diff, short *action, int* alive_agent, int board_size);
__global__ void alive_agent_reset(int* alive_agent, int* total_rewards, int2* flag_table);
__global__ void epsilon_reset(float* agent_epsilon, float epsilon);
__global__ void agent_epsilon_update(float* agent_epsilon);

void agent_init()
{
	min_epsilon = 0.03;
	learning_rate = 0.01;
	global_start = true;
	epsilon = 1.0;

	// Init Qtable once
	int dimx = 32;
	int dimy = 32;
	dim3 block(dimx, dimy);
	dim3 grid((BOARD_SIZE * 2 + block.x - 1) / block.x, (BOARD_SIZE * 2 + block.y - 1) / block.y);

	curandState_t* rand_states;
	checkCudaErrors(cudaMalloc((void**)&rand_states, 4 * BOARD_SIZE * BOARD_SIZE * sizeof(curandState_t)));
	q_table_rand_init << <grid, block >> >(1377, rand_states, BOARD_SIZE);

	checkCudaErrors(cudaMalloc((void**)&d_Qtable, sizeof(float) * 4 * BOARD_SIZE * BOARD_SIZE));
	q_table_init_cuda_random << <grid, block >> > (rand_states, d_Qtable, BOARD_SIZE);
	cudaFree(rand_states);

	checkCudaErrors(cudaMalloc((void**)&d_action, sizeof(short)*XCEIL(NUM_AGENT, 5)));
	checkCudaErrors(cudaMalloc((void**)&d_Qtabla_diff, sizeof(float) * 4 * BOARD_SIZE * BOARD_SIZE));
	checkCudaErrors(cudaMalloc((void**)&d_Alive_agent, sizeof(int) *XCEIL(NUM_AGENT, 5)));
	checkCudaErrors(cudaMalloc((void**)&d_total_rewards, sizeof(int) *XCEIL(NUM_AGENT, 5)));
	checkCudaErrors(cudaMalloc((void**)&d_flag_table, sizeof(int2)*NUM_AGENT*FLAG_TARGET));
	//policy
	checkCudaErrors(cudaMalloc((void**)&d_agent_epsilon, sizeof(float) * sizeof(int) *XCEIL(NUM_AGENT, 5)));


	dim3 nthInBlk;
	dim3 nblkInGrid;
	nthInBlk.x = std::min(NUM_AGENT, (1 << LOG2_AGENT_STRIDE));
	nthInBlk.y = 1;
	nthInBlk.z = 1;
	nblkInGrid.x = SCEIL(NUM_AGENT, LOG2_AGENT_STRIDE);
	nblkInGrid.y = 1;
	nblkInGrid.z = 1;

	cudaMalloc((void**)&rand_states_action, XCEIL(NUM_AGENT, 5) * sizeof(curandState_t));
	rand_init << <nblkInGrid, nthInBlk >> >(1377, rand_states_action);
	epsilon_reset << <nblkInGrid, nthInBlk >> >(d_agent_epsilon, epsilon);


	alive_agent_reset << <nblkInGrid, nthInBlk >> >(d_Alive_agent, d_total_rewards, d_flag_table);
	return;
}
void agent_clearaction()
{
	dim3 nthInBlk;
	dim3 nblkInGrid;

	nthInBlk.x = std::min(NUM_AGENT, (1 << LOG2_AGENT_STRIDE));
	nthInBlk.y = 1;
	nthInBlk.z = 1;

	nblkInGrid.x = SCEIL(NUM_AGENT, LOG2_AGENT_STRIDE);
	nblkInGrid.y = 1;
	nblkInGrid.z = 1;

	if (!global_start)
	{
		checkCudaErrors(cudaMemcpyAsync(total_rewards, d_total_rewards, sizeof(int) * NUM_AGENT, cudaMemcpyDeviceToHost));
		int total = 0;
		int flag_total = 0;
		for (int i = 0;i < NUM_AGENT;i++)
		{
			total += total_rewards[i];
			if (total_rewards[i] == FLAG_TARGET)
				flag_total++;

		}
		// To check the performance
		printf("....Average Rewards %2.4f, # of best agents : %d\n", total / (1.0*NUM_AGENT), flag_total);

	}
	else
	{
		global_start = false;
	}

	epsilon_reset << <nblkInGrid, nthInBlk >> >(d_agent_epsilon, epsilon);
	alive_agent_reset << <nblkInGrid, nthInBlk >> >(d_Alive_agent, d_total_rewards, d_flag_table);

	return;
}

float agent_adjustepsilon()
{
	if (epsilon > min_epsilon)
	{
		epsilon *= 0.998;
	}
	else if (epsilon < 0.9)
	{
		epsilon *= 0.995;
	}
	else if (epsilon < 0.08)
	{
		epsilon = 0.00;
		//learning_rate *= 5.0;
	}

	dim3 nthInBlk;
	dim3 nblkInGrid;

	nthInBlk.x = std::min(NUM_AGENT, (1 << LOG2_AGENT_STRIDE));
	nthInBlk.y = 1;
	nthInBlk.z = 1;

	nblkInGrid.x = SCEIL(NUM_AGENT, LOG2_AGENT_STRIDE);
	nblkInGrid.y = 1;
	nblkInGrid.z = 1;

	agent_epsilon_update << <nblkInGrid, nthInBlk >> >(d_agent_epsilon);

	return epsilon;
}

short* agent_action(int2* cstate)
{
	dim3 nthInBlk;
	dim3 nblkInGrid;

	nthInBlk.x = std::min(NUM_AGENT, (1 << LOG2_AGENT_STRIDE));
	nthInBlk.y = 1;
	nthInBlk.z = 1;

	nblkInGrid.x = SCEIL(NUM_AGENT, LOG2_AGENT_STRIDE);
	nblkInGrid.y = 1;
	nblkInGrid.z = 1;

	agent_action_cuda_random << <nblkInGrid, nthInBlk >> > (cstate, rand_states_action, d_action, d_Qtable, d_Alive_agent, d_total_rewards,
		d_agent_epsilon, BOARD_SIZE);
	return d_action;
}
void agent_update(int2* cstate, int2* nstate, float *rewards)
{
	dim3 nthInBlk;
	dim3 nblkInGrid;

	nthInBlk.x = std::min(NUM_AGENT, (1 << LOG2_AGENT_STRIDE));
	nthInBlk.y = 1;
	nthInBlk.z = 1;

	nblkInGrid.x = SCEIL(NUM_AGENT, LOG2_AGENT_STRIDE);
	nblkInGrid.y = 1;
	nblkInGrid.z = 1;

	float q_table[4 * BOARD_SIZE*BOARD_SIZE];
	checkCudaErrors(cudaMemcpyAsync(q_table, d_Qtable, sizeof(float) * 4 * BOARD_SIZE * BOARD_SIZE, cudaMemcpyDeviceToHost));
	agent_cal_diff_cuda << <nblkInGrid, nthInBlk >> > (cstate, nstate, rewards, d_Qtable, d_Qtabla_diff, d_action, d_Alive_agent, d_total_rewards, d_flag_table, d_agent_epsilon, learning_rate);
	agent_update_qtable_cuda << <nblkInGrid, nthInBlk >> > (cstate, nstate, rewards, d_Qtable, d_Qtabla_diff, d_action, d_Alive_agent, BOARD_SIZE);
	return;
}

/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
__global__ void epsilon_reset(float* agent_epsilon, float epsilon)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	agent_epsilon[tid] = epsilon;
	return;
}

__global__ void agent_epsilon_update(float* agent_epsilon)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	agent_epsilon[tid] *= 0.995;

	if (agent_epsilon[tid] < 0.08)
	{
		agent_epsilon[tid] = 0.00;
	}
	return;
}

__global__ void q_table_init_cuda_random(curandState_t* states, float* qtable, int board_size)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * 2 * board_size + ix;

	unsigned int action = idx / (board_size*board_size);
	idx = idx % (board_size*board_size);
	unsigned int pos_y = idx / board_size;
	unsigned int pos_x = idx % board_size;
	idx = iy * 2 * board_size + ix;

	if (ix < board_size * 2 && iy < board_size * 2)
	{
		curand_init(clock() + idx, idx, 0, &states[idx]);
		float rand_qvalue = (curand_uniform(&(states[idx]))) / 200.0;

		if (pos_x == 0 && action == 2)
			rand_qvalue = 0.0;
		if (pos_x == board_size - 1 && action == 0)
			rand_qvalue = 0.0;
		if (pos_y == 0 && action == 3)
			rand_qvalue = 0.0;
		if (pos_y == board_size - 1 && action == 1)
			rand_qvalue = 0.0;
		qtable[idx] = rand_qvalue;
	}
}


__global__ void q_table_rand_init(unsigned int seed, curandState_t* states, int board_size)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * board_size * 2 + ix;

	if (ix < 2 * board_size && iy < 2 * board_size)
		curand_init(clock() + idx, idx, 0, &states[idx]);
}

__global__ void alive_agent_reset(int* alive_agent, int* total_rewards, int2* flag_table)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	alive_agent[tid] = 1;
	total_rewards[tid] = 0;

	for (int i = 0;i < FLAG_TARGET;i++)
	{
		flag_table[tid*FLAG_TARGET + i].x = -1;
		flag_table[tid*FLAG_TARGET + i].y = -1;
	}

}


__global__ void agent_action_cuda_random(int2* cstate, curandState_t* states, short *action, float* qtable
	, int* alive_agent, int* total_rewards, float* agent_epsilon, int board_size)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	float rand_greedy = curand_uniform(&(states[tid]));
	float rand_action = curand_uniform(&(states[tid])) - 0.0001;
	int2 agent = cstate[tid];

	int pos = agent.x + agent.y * board_size;
	int max_action;
	float next_q_max = -30.0;
	float next_q;

	if (alive_agent[tid] == 0)
	{
		action[tid] = -1;
		if (total_rewards[tid] == FLAG_TARGET)
		{
			action[tid] = -2;
		}
		return;
	}



	//if (rand_greedy > epsilon && next_q_max > 0.0)
	//if ((epsilon < 0.2 && next_q_max > 0.0) || (rand_greedy > epsilon && next_q_max > -0.1))
	if (rand_greedy > agent_epsilon[tid])
	{
		for (int i = 0;i < 4;i++)
		{
			int next_qtable_pos = i * board_size* board_size + pos;
			next_q = qtable[next_qtable_pos];
			if (next_q > next_q_max)
			{
				max_action = i;
				next_q_max = next_q;
			}
		}
		action[tid] = max_action;
	}
	else
	{
		action[tid] = rand_action * 4;
		//printf("%d agent moves %d at (%d, %d) random \n", tid, action[tid], agent.x, agent.y);
	}
}

__global__ void rand_init(unsigned int seed, curandState_t* states)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(clock() + tid, tid, 0, &states[tid]);
}


__global__ void agent_cal_diff_cuda(int2* cstate, int2* nstate, float *rewards, float *qtable, float* qtabld_diff, short *action, int* alive_agent, int* total_rewards,
	int2* flag_table, float* agent_epsilon,  float learning_rate)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	// updata q table
	int2 agent = cstate[tid];
	int2 agent_next = nstate[tid];
	// read board status to identify agent alive
	int pos = agent.x + agent.y*BOARD_SIZE;
	int new_pos = agent_next.x + agent_next.y*BOARD_SIZE;
	short reward = rewards[tid];
	int qtable_pos = action[tid] * BOARD_SIZE* BOARD_SIZE + pos;
	int max_action;
	float next_q_max = -30.0;
	float next_q;
	for (int i = 0;i < 4;i++)
	{
		int next_qtable_pos = i * BOARD_SIZE* BOARD_SIZE + new_pos;
		next_q = qtable[next_qtable_pos];
		if (next_q > next_q_max)
		{
			max_action = i;
			next_q_max = next_q;
		}
	}

	if (alive_agent[tid] == 1)
	{
		if (reward == 0)
		{

			qtabld_diff[qtable_pos] += learning_rate * (0.9 * next_q_max - qtable[qtable_pos]);
		}
		else if (reward > 0)
		{
			bool visited = false;
			for (int i = 0;i < total_rewards[tid];i++)
			{
				if (flag_table[tid*FLAG_TARGET + i].x == agent_next.x && flag_table[tid*FLAG_TARGET + i].y == agent_next.y)
				{
					visited = true;
					break;
				}
			}
			if (!visited)
			{
				agent_epsilon[tid] *= 1.7;
				flag_table[tid*FLAG_TARGET + total_rewards[tid]].x = agent_next.x;
				flag_table[tid*FLAG_TARGET + total_rewards[tid]].y = agent_next.y;
				total_rewards[tid] += 1;

				qtabld_diff[qtable_pos] += learning_rate * (total_rewards[tid] * 100 + 0.9 * next_q_max - qtable[qtable_pos]);
				if (total_rewards[tid] == FLAG_TARGET)
					alive_agent[tid] = 0;
			}

		}
		else
		{
			qtabld_diff[qtable_pos] += learning_rate * (-1.0 - qtable[qtable_pos]);
			alive_agent[tid] = 0;
		}
	}
}

__global__ void agent_update_qtable_cuda(int2* cstate, int2* nstate, float *rewards, float *qtable, float* qtabld_diff, short *action, int* alive_agent, int board_size)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	// updata q table
	int2 agent = cstate[tid];
	int2 agent_next = nstate[tid];
	// read board status to identify agent alive
	int pos = agent.x + agent.y*board_size;
	int new_pos = agent_next.x + agent_next.y*board_size;
	int qtable_pos = action[tid] * board_size* board_size + pos;
	if (pos != new_pos)
	{
		qtable[qtable_pos] += qtabld_diff[qtable_pos];
	}
	qtabld_diff[qtable_pos] = 0;
}