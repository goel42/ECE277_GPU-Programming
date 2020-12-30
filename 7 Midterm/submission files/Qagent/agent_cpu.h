/*************************************************************************
/* ECE 277: GPU Programmming 2020 FALL quarter
/* Author and Instructer: Cheolhong An
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
#include <cuda_runtime.h>

#pragma once

void agent_init();
void agent_init_episode();
float agent_adjustepsilon();
short* agent_action(int2* cstate);
void agent_update(int2* cstate, int2* nstate, float *rewards);