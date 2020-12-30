#pragma once
#include <chrono>
#include <iomanip> 
#include <sstream>


#define  BOARD_SIZE 46 
#define  MINE_COUNT BOARD_SIZE*3

#define  FLAG_COUNT 80 //5, 10, 15, 50
#define  AGENT_CODE 0x2
#define  AGENT_MASK 0xfffffffd
#define  MINE 0x80000000
#define  FLAG  1 //5, 10, 15, 20
#define MINE_AGENT 0x80000002 
#define FLAG_AGENT 0x00000003 
