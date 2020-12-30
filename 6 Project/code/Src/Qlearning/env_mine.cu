#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <curand.h>
#include <curand_kernel.h>

#include "common_def.h"
#include "env_mine.h"
#include "util_fun.h"
#include "draw_env.h"

#ifdef _DEBUG
#define __FORCEINLINE__ 
#else
#define __FORCEINLINE__ __forceinline__
#endif 


//////////////////////////////////////////////////////////////////
__global__ void assign_randomposition(int2* state, float* rewards, int maxVal);
__global__ void update_board(int2* state, int *board, int board_width);
__global__ void clear_board(int2* state, int *board, int board_width, float* rewards, bool* flag_board);
__global__ void env_step(int2* cstate, int2* nstate,
	float* rewards, short* actions, int *board, int board_width, int board_height, bool* flag_board);
//////////////////////////////////////////////////////////////////

env_mineCls::~env_mineCls()
{
	if (d_board != NULL) checkCudaErrors(cudaFree(d_board));
	if (d_state[0] != NULL) checkCudaErrors(cudaFree(d_state[0]));
	if (d_state[1] != NULL) checkCudaErrors(cudaFree(d_state[1]));
	if (d_reward != NULL) checkCudaErrors(cudaFree(d_reward));
	if (m_state != NULL) free(m_state);
}

void env_mineCls::init(int boardsize, int* board, int num_agent)
{
	m_board_stride = XCEIL(boardsize, 5);
	m_board_height = boardsize;
	m_board_width = boardsize;

	m_num_agent = num_agent;

	m_state = (int2*)malloc(sizeof(int2)*XCEIL(num_agent, 5));

	checkCudaErrors(cudaMalloc((void**)&d_board, sizeof(int)*m_board_stride*m_board_height));

	checkCudaErrors(cudaMalloc((void**)&d_flag_board, sizeof(bool)*m_board_stride*m_board_height));

	checkCudaErrors(cudaMalloc((void**)&d_state[0], sizeof(int2)*XCEIL(num_agent, 5)));
	checkCudaErrors(cudaMalloc((void**)&d_state[1], sizeof(int2)*XCEIL(num_agent, 5)));

	checkCudaErrors(cudaMalloc((void**)&d_reward, sizeof(float)*XCEIL(num_agent, 5)));

	checkCudaErrors(cudaMemcpyAsync(d_board, board, sizeof(int)*m_board_width*m_board_height, cudaMemcpyHostToDevice));

};

void env_mineCls::reset(int sid)
{
	dim3 nthInBlk;
	dim3 nblkInGrid;

	nthInBlk.x = std::min(m_num_agent, (1 << LOG2_AGENT_STRIDE));
	nthInBlk.y = 1;
	nthInBlk.z = 1;

	nblkInGrid.x = SCEIL(m_num_agent, LOG2_AGENT_STRIDE);
	nblkInGrid.y = 1;
	nblkInGrid.z = 1;
	// clear agent position
	clear_board << <nblkInGrid, nthInBlk >> > (d_state[sid], d_board, m_board_width, d_reward, d_flag_board);

	assign_randomposition << <nblkInGrid, nthInBlk >> > (d_state[sid], d_reward, m_board_height);

	update_board << <nblkInGrid, nthInBlk >> > (d_state[sid], d_board, m_board_width);
}

void env_mineCls::clearboard(int sid)
{
	dim3 nthInBlk;
	dim3 nblkInGrid;

	nthInBlk.x = std::min(m_num_agent, (1 << LOG2_AGENT_STRIDE));
	nthInBlk.y = 1;
	nthInBlk.z = 1;

	nblkInGrid.x = SCEIL(m_num_agent, LOG2_AGENT_STRIDE);
	nblkInGrid.y = 1;
	nblkInGrid.z = 1;
	// clear agent position
	clear_board << <nblkInGrid, nthInBlk >> > (d_state[sid], d_board, m_board_width, d_reward, d_flag_board);
}

void env_mineCls::step(int sid, short* actions)
{
	dim3 nthInBlk;
	dim3 nblkInGrid;

	nthInBlk.x = std::min(m_num_agent, (1 << LOG2_AGENT_STRIDE));
	nthInBlk.y = 1;
	nthInBlk.z = 1;

	nblkInGrid.x = SCEIL(m_num_agent, LOG2_AGENT_STRIDE);
	nblkInGrid.y = 1;
	nblkInGrid.z = 1;

	env_step << <nblkInGrid, nthInBlk >> > (d_state[sid], d_state[sid^1], d_reward, actions, d_board, m_board_width, m_board_height, d_flag_board);
}

void env_mineCls::render(int* board, int sid)
{
	// copy gpu to cpu memory
	checkCudaErrors(cudaMemcpy(board, d_board, sizeof(int)*m_board_height*m_board_width, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(m_state, d_state[sid], sizeof(int2)*m_num_agent, cudaMemcpyDeviceToHost));
}

/////////////////////////////////////////////////////////////////
__global__ void assign_randomposition(int2* state, float *rewards, int maxVal)
{
	// assign random x,y position with univorm distribution from 0 to 1.0
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	curandState state_p, state_s;
	curand_init(clock() + tid, tid, 0, &state_p);
	curand_init(clock() + tid, tid, 0, &state_s);

	int2 node;
	node.x = curand_uniform(&state_p)*maxVal;
	node.y = curand_uniform(&state_s)*maxVal;
	//node.x = 0;
	//node.y = 0;

	state[tid] = node;
	rewards[tid] = 0.0f;
}

__global__ void clear_board(int2* state, int *board, int board_width, float* rewards, bool* flag_board)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int2 agent = state[tid];
	int pos = agent.x + agent.y*board_width;
	int tmp = board[pos];
	board[pos] &= AGENT_MASK; // set agent position.
	rewards[tid] = 0.0f;
	flag_board[tid] = false;
	//printf("%d, (%d,%d, %x, %x)\n", tid, agent.x, agent.y, tmp, board[pos]);
}

__global__ void update_board(int2* state, int *board, int board_width)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int2 agent = state[tid];
	int pos = agent.x + agent.y*board_width;
	board[pos] |= AGENT_CODE; // set agent position.

}

__FORCEINLINE__ __device__ void clearboard(int2 agent, int* board, int board_width)
{
	int pos = agent.x + agent.y*board_width;
	board[pos] &= AGENT_MASK; // set agent position.
}

__FORCEINLINE__ __device__ int reward_updateboard(int2 agent, int* board, int board_width)
{
	int pos = agent.x + agent.y*board_width;
	int reward = board[pos];
	board[pos] = reward | AGENT_CODE; // set agent position.
	
	return reward;
}

__global__ void env_step(int2* cstate, int2* nstate,
	float* rewards, short* actions, int *board, int board_width, int board_height, bool* flag_board)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int2 agent = cstate[tid];
	short action = actions[tid];
	// read board status to identify agent alive
	int pos = agent.x + agent.y*board_width;
	int boardstatus = board[pos];

	//int alive = ((boardstatus == MINE_AGENT) || (boardstatus == FLAG_AGENT)) ? 0 : 1;
	int alive = (boardstatus == MINE_AGENT || actions[tid] < 0) ? 0 : 1;
	if (actions[tid] == -2)
	{
		flag_board[pos] = true;
	}
	
	if (alive) { // alive
		// clear board information
		board[pos] = boardstatus & AGENT_MASK;
		//clearboard(agent, board, board_width);
		// 0: right, 1: bottom , 2: left, 3: top
		int code = action & 0x3;
		int code1 = action & 0x1;
		int dx = 0; 
		int dy = 0;
		dx = (~code1 & 0x1);
		dy = (code1 & 0x1);
		dx = (code > 1) ? -dx : dx;
		dy = (code > 1) ? -dy : dy;

		agent.x = min(board_width - 1, agent.x + dx);
		agent.x = max(0, agent.x);

		agent.y = min(board_height - 1, agent.y + dy);
		agent.y = max(0, agent.y);

		// check reward and update board
		int reward = reward_updateboard(agent, board, board_width);
		rewards[tid] = (reward & MINE) ? -1 : reward;
		
		if (flag_board[pos])
		{
			board[pos] = FLAG_AGENT;
		}

	}
	else {
		rewards[tid] = -1;
	}

	nstate[tid] = agent;
}