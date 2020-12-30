#include <curand.h>
#include <curand_kernel.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "qlearning.h"
#include "draw_env.h"
#include "common_def.h"

extern void agent_init();
extern void agent_clearaction();
extern float agent_adjustepsilon();
extern short* agent_action(int2* cstate);
extern void agent_update(int2* cstate, int2* nstate, float *rewards);
/////////////////////////////////////////////////////////
extern unsigned int active_agent;
extern unsigned int flag_agent;

qlearningCls::~qlearningCls()
{

}

void qlearningCls::init(int boardsize, int *board)
{
	env.init(boardsize, board, NUM_AGENT);

	m_sid = 0;
	m_episode = 0;
	m_steps = 0;
	m_newepisode = false;
	m_boardsize = boardsize;

#ifdef _DEBUG
#define MByte (1024*1024)
	size_t remainMem, totalMem;
	checkCudaErrors(cudaMemGetInfo(&remainMem, &totalMem));
	printf("GPU total memoroy: %d MB, remaining memory: %d MB\n", (totalMem / MByte), (remainMem / MByte));
#endif
}

int qlearningCls::checkstatus(int* board, int2* state, unsigned int &fagent)
{
	int2 agent;
	int pos;
	int alive_agent = 0;
	int flag_agent = 0;
	int mine_agent = 0;
	for (int k = 0; k < NUM_AGENT; k++) {
		agent = state[k];
		pos = agent.x + agent.y*m_boardsize;
		switch (board[pos]) {
			case AGENT_CODE:
				alive_agent++;
				break;
			case FLAG_AGENT:
			case FLAG:
				flag_agent++;
				break;
			case MINE_AGENT:
			case MINE:
				mine_agent++;
				break;
		}
	}

	fagent = flag_agent;

	if (mine_agent >= (NUM_AGENT*0.65)) {
		m_newepisode = true;
	}
	else {
		m_newepisode = false;
	}

	return alive_agent;
}

int qlearningCls::alive_agent(int* board)
{
	// check number of survival
	int alive_agent = 0;
	for (int k = 0; k < m_boardsize*m_boardsize; k++) {
		if (board[k] == AGENT_CODE) {
			alive_agent++;
		}
	}

	if (alive_agent <= (NUM_AGENT*0.2)) {
		m_newepisode = true;
	}
	else {
		m_newepisode = false;
	}
	return alive_agent;
}

int qlearningCls::learning(int *board, unsigned int &episode, unsigned int &steps)
{
	if (m_episode == 0 && m_steps==0) {// only for first episode
		env.reset(m_sid);
		agent_init(); // clear action + initQ table + self initialization
	}
	else {
		active_agent = checkstatus(board, env.m_state, flag_agent);
		
		if (m_newepisode) {
			env.reset(m_sid); // clear buffer
			float epsilon = agent_adjustepsilon(); // adjust epsilon
			agent_clearaction(); // clear action

			m_steps = 0;
			printf("Episode=%4d, epsilon=%4.3f\n", m_episode, epsilon);
			m_episode++;
		}
		else {
			short* d_action = agent_action(env.d_state[m_sid]);
			env.step(m_sid, d_action); // current state buffer
			agent_update(env.d_state[m_sid], env.d_state[m_sid ^ 1], env.d_reward);
			m_sid ^= 1;
			episode = m_episode;
			steps = m_steps;
			
		}

	}
	m_steps++;
	env.render(board, m_sid);
	return m_newepisode;
}