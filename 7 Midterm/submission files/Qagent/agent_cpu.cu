/*************************************************************************
/* ECE 277: GPU Programmming 2020 FALL quarter
/* Author and Instructer: Cheolhong An
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "agent.h"
#include "agent_cpu.h"

Agent *agents;

void agent_init() {
	// clear action + init Q table + self initialization
	// update parameters for every single agent
	agents = new Agent;
	agents->init_New();
}

void agent_init_episode() {
	// set all agents in active status, as agent initialization
	agents->init_Agents();
}

float agent_adjustepsilon() {
	// adjust epsilon
	return agents->update_Epsilon();
}

short* agent_action(int2* cstate) {
	agents->action_Taken(cstate);
	return agents->result_Actions();
}

void agent_update(int2* cstate, int2* nstate, float *rewards) {
	agents->update_Agents(cstate,nstate,rewards);
}