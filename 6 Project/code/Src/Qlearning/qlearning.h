#pragma once

#include "env_mine.h"
#include "common_def.h"

class qlearningCls
{
public:
	~qlearningCls();
	void init(int boardsize, int *board);
	int learning(int *board, unsigned int &episode, unsigned int &steps);
	int alive_agent(int* board);
	int checkstatus(int* board, int2* state, unsigned int &fagent);
private:
	env_mineCls env;
	int m_sid;
	bool m_newepisode;
	unsigned int m_episode;
	unsigned int m_steps;
	int m_boardsize;
	
};