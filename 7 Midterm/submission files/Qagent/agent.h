/*************************************************************************
/* ECE 277: GPU Programmming 2020 FALL quarter
/* Author and Instructer: Cheolhong An
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
//Defines the data structure for the agent

#pragma once
#include <curand_kernel.h>

class Agent {
private:
	
	float* d_qtable;
	short *d_action;
	bool *d_alive;
	float epsilon;
	float *d_epsilon;
	
	curandState *randState;
	
	// qLearning Paramters
	float alpha;		// learning rate
	float gamma;		// discount factor

	// define grid and block
	dim3 block;
	dim3 grid;

	void init_Qtable();
	void init_Variables();

public:
	~Agent() {
		cudaFree(randState);
		cudaFree(d_epsilon);
		cudaFree((void *)d_qtable);
	}
	void init_New();
	void init_Agents();
	float update_Epsilon();
	void action_Taken(int2* cstate);
	void update_Agents(int2* cstate, int2* nstate, float *rewards);
	short* result_Actions() {
		return this->d_action;
	}
};