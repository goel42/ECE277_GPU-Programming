#include <cmath>
#include <random>

#include <string>
////////////////////////////////////////////////////////////////////////////
//// modified Simple Minesweeper using OpenGL / GLUT codes
///https://codereview.stackexchange.com/questions/158957/simple-minesweeper-using-opengl-glut
////////////////////////////////////////////////////////////////////////////
#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined (__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

#include "draw_env.h"
//////////////////////////////////////


enum { AGENT = AGENT_CODE};

enum { TILE_SIZE = 20 };
enum { MARGIN = 40 };
enum { PADDING = 10 };

unsigned int window_width = BOARD_SIZE*TILE_SIZE + 2 * PADDING;
unsigned int window_height = BOARD_SIZE*TILE_SIZE + 2 * PADDING;

extern int new_episode;

enum Color {
	RED,
	DARKRED,
	BLUE,
	DARKBLUE,
	GREEN,
	DARKGREEN,
	CYAN,
	DARKCYAN,
	YELLOW,
	DARKYELLOW,
	WHITE,
	MAGENTA,
	BLACK,
	DARKGRAY,
	LIGHTGRAY,
	ULTRALIGHTGRAY
};



struct cell
{
	int type;
};

int board[BOARD_SIZE*BOARD_SIZE];
static const struct
{
	float r, g, b;
} colors[] =
{
	{ 1, 0, 0 },// red
	{ 0.5f, 0, 0 },// dark red

	{ 0, 0, 1 }, // blue
	{ 0, 0, 0.5f }, // dark blue

	{ 0, 1, 0 }, // green
	{ 0, 0.5f, 0 }, // dark green

	{ 0, 1, 1 }, // cyan
	{ 0, 0.5f, 0.5f }, // dark  cyan

	{ 1, 1, 0 },//yellow
	{ 0.5f, 0.5f, 0 },//dark yellow

	{ 1, 1, 1 },// White
	{ 1, 0, 1 }, // magenta

	{ 0, 0, 0 }, // black
	{ 0.25, 0.25, 0.25 }, // dark gray
	{ 0.5, 0.5, 0.5 }, // light gray
	{ 0.75, 0.75, 0.75 }, // ultra-light gray

};

void drawRect(int x, int y, float width, float height, const Color& color = LIGHTGRAY, bool outline = true)
{
	glColor3f(colors[color].r, colors[color].g, colors[color].b);
	glBegin(outline ? GL_LINE_STRIP : GL_TRIANGLE_FAN);
	{   
		glVertex2i(x + 0 * width, y + 0 * height);
		glVertex2i(x + 1 * width, y + 0 * height);
		glVertex2i(x + 1 * width, y + 1 * height);
		glVertex2i(x + 0 * width, y + 1 * height);
	}
	glEnd();
}

void drawFrame(float x, float y, float width, float height, bool doubleFrame = true)
{

	glColor3f(colors[WHITE].r, colors[WHITE].g, colors[WHITE].b);
	glBegin(GL_LINE_LOOP);
	{
		glVertex2f((x + 0) + 0 * width, (y - 0) + 0 * height);
		glVertex2f((x - 0) + 0 * width, (y - 1) + 1 * height);
		glVertex2f((x - 1) + 1 * width, (y - 1) + 1 * height);
		glVertex2f((x - 2) + 1 * width, (y - 2) + 1 * height);
		glVertex2f((x + 1) + 0 * width, (y - 2) + 1 * height);
		glVertex2f((x + 1) + 0 * width, (y + 1) + 0 * height);
	}
	glEnd();

	glColor3f(colors[LIGHTGRAY].r, colors[LIGHTGRAY].g, colors[LIGHTGRAY].b);
	glBegin(GL_LINE_LOOP);
	{
		glVertex2f((x - 2) + 1 * width, (y - 2) + 1 * height);
		glVertex2f((x - 2) + 1 * width, (y + 1) + 0 * height);
		glVertex2f((x + 1) + 0 * width, (y + 1) + 0 * height);
		glVertex2f((x - 0) + 0 * width, (y - 0) + 0 * height);
		glVertex2f((x - 1) + 1 * width, (y - 0) + 0 * height);
		glVertex2f((x - 1) + 1 * width, (y - 1) + 1 * height);
	}
	glEnd();

	if (!doubleFrame) return;

	width = width - 2 * PADDING;
	height = height - 2 * PADDING;


	glBegin(GL_LINE_LOOP);
	{
		glVertex2f((x - 0 + PADDING) + 0 * width, (y + PADDING - 0) + 0 * height);
		glVertex2f((x - 0 + PADDING) + 0 * width, (y + PADDING - 1) + 1 * height);
		glVertex2f((x - 1 + PADDING) + 1 * width, (y + PADDING - 1) + 1 * height);
		glVertex2f((x - 2 + PADDING) + 1 * width, (y + PADDING - 2) + 1 * height);
		glVertex2f((x + 1 + PADDING) + 0 * width, (y + PADDING - 2) + 1 * height);
		glVertex2f((x + 1 + PADDING) + 0 * width, (y + PADDING + 1) + 0 * height);
	}
	glEnd();
	glColor3f(colors[WHITE].r, colors[WHITE].g, colors[WHITE].b);

	glBegin(GL_LINE_LOOP);
	{
		glVertex2i((x + PADDING - 2) + 1 * width, (y + PADDING - 2) + 1 * height);
		glVertex2i((x + PADDING - 2) + 1 * width, (y + PADDING + 1) + 0 * height);
		glVertex2i((x + PADDING + 1) + 0 * width, (y + PADDING + 1) + 0 * height);
		glVertex2i((x + PADDING - 0) + 0 * width, (y + PADDING - 0) + 0 * height);
		glVertex2i((x + PADDING - 1) + 1 * width, (y + PADDING - 0) + 0 * height);
		glVertex2i((x + PADDING - 1) + 1 * width, (y + PADDING - 1) + 1 * height);
	}
	glEnd();
}

void drawUpperFrame(int x = 0, int y = 0)
{
	static const float upper_frame_outter_width = window_width;
	static const float upper_frame_outter_height = 2 * MARGIN;
	static const float offset = window_height - upper_frame_outter_height;

	drawFrame(0, offset, upper_frame_outter_width, upper_frame_outter_height);
}

void drawLowerFrame(int x = 0, int y = 0)
{
	static const float lower_frame_outter_size = window_width;
	drawFrame(0, 0, lower_frame_outter_size, lower_frame_outter_size);
}

int index(int x, int y)
{
	return x + (y*BOARD_SIZE);
}
int getType(int x, int y)
{
	return board[index(x, y)];
}

void setType(int x, int y, int v)
{
	board[index(x, y)] = v;
}

void drawClosedDim(int x, int y)
{
	drawFrame(x *TILE_SIZE + PADDING, y*TILE_SIZE + PADDING, TILE_SIZE, TILE_SIZE, false);
}

void drawOpenDim(int x, int y, const Color& color = LIGHTGRAY, bool outline = true)
{
	drawRect(x*TILE_SIZE + PADDING, y*TILE_SIZE + PADDING, TILE_SIZE, TILE_SIZE, color, outline);
}

void drawFlag(int x, int y)
{
	glColor3f(colors[BLACK].r, colors[BLACK].g, colors[BLACK].b);
	x = (x*TILE_SIZE) + PADDING + 6;
	y = (y*TILE_SIZE) + PADDING + 3;

	//platform
	glBegin(GL_POLYGON);
	{
		glVertex2i(x + 0, y + 2);
		glVertex2i(x + 9, y + 2);
		glVertex2i(x + 9, y + 3);
		glVertex2i(x + 7, y + 3);
		glVertex2i(x + 7, y + 4);
		glVertex2i(x + 3, y + 4);
		glVertex2i(x + 3, y + 3);
		glVertex2i(x + 0, y + 3);
	}
	glEnd();

	//mast
	glBegin(GL_LINES);
	{
		glVertex2i(x + 4, y + 4);
		glVertex2i(x + 4, y + 7);
	}
	glEnd();

	//flag
	glColor3f(colors[RED].r, colors[RED].g, colors[RED].b);
	glBegin(GL_TRIANGLES);
	{
		glVertex2i(x + 5, y + 7);
		glVertex2i(x + 5, y + 12);
		glVertex2i(x + 0, y + 9);
	}
	glEnd();
}

void drawMine(int x, int y, bool dead)
{
	if (dead)
	{
		drawRect(x*TILE_SIZE + PADDING, y*TILE_SIZE + PADDING, TILE_SIZE, TILE_SIZE, RED, false);
	}


	x = (x*TILE_SIZE) + PADDING + 4;
	y = (y*TILE_SIZE) + PADDING + 4;

	//spikes
	glColor3f(colors[BLACK].r, colors[BLACK].g, colors[BLACK].b);
	glBegin(GL_LINES);
	{
		glVertex2i(x + 5, y - 1);
		glVertex2i(x + 5, y + 12);

		glVertex2i(x - 1, y + 5);
		glVertex2i(x + 12, y + 5);

		glVertex2i(x + 1, y + 1);
		glVertex2i(x + 10, y + 10);

		glVertex2i(x + 1, y + 10);
		glVertex2i(x + 10, y + 1);
	}
	glEnd();

	//ball
	glBegin(GL_POLYGON);
	{
		glVertex2i(x + 3, y + 1);
		glVertex2i(x + 1, y + 4);
		glVertex2i(x + 1, y + 7);
		glVertex2i(x + 3, y + 10);
		glVertex2i(x + 8, y + 10);
		glVertex2i(x + 10, y + 7);
		glVertex2i(x + 10, y + 4);
		glVertex2i(x + 8, y + 1);
	}
	glEnd();

	//shine
	drawRect(x + 3, y + 5, 2, 2, WHITE, false);
}

void drawOpen(int x, int y, int n)
{
	switch (n) {
	case 0:
		drawOpenDim(x, y);
		break;
	case MINE:
		drawOpenDim(x, y, LIGHTGRAY, true);
		drawOpenDim(x, y);
		drawMine(x, y, false);
		break;
	case MINE_AGENT:
		drawMine(x, y, true);
		break;
	case FLAG:
		drawFlag(x, y);
		drawOpenDim(x, y);
		break;
	case FLAG_AGENT:
		drawOpenDim(x, y, BLUE, false);
		drawFlag(x, y);
		break;
	case AGENT:
		drawOpenDim(x, y, GREEN, false);
		drawOpenDim(x, y);
		break;
	default:
		drawOpenDim(x, y);
	}
}


void draw()
{
	drawLowerFrame();

	int code;

	for (int y = 0; y < BOARD_SIZE; y++)
	{
		for (int x = 0; x < BOARD_SIZE; x++)
		{
			code = getType(x, y);
			drawOpen(x, (BOARD_SIZE-y-1), code);
		}
	}
}



bool isMine(int x, int y)
{
	if (x < 0 || y < 0 || x > BOARD_SIZE - 1 || y > BOARD_SIZE - 1)
		return false;

	if (getType(x, y) == MINE)
		return true;
	return false;
}

int calcMine(int x, int y)
{
	return isMine(x - 1, y - 1)
		+ isMine(x, y - 1)
		+ isMine(x + 1, y - 1)
		+ isMine(x - 1, y)
		+ isMine(x + 1, y)
		+ isMine(x - 1, y + 1)
		+ isMine(x, y + 1)
		+ isMine(x + 1, y + 1);
}

int rand_int(int low, int high)
{
	static std::default_random_engine re{ std::random_device{}() };
	using Dist = std::uniform_int_distribution<int>;
	static Dist uid{};
	return uid(re, Dist::param_type{ low,high });
}

static std::vector<int> generate_data(size_t size, int low, int high, bool fixed=false)
{
	using value_type = int;
	static std::uniform_int_distribution<value_type> distribution(
		low, high);

	std::vector<value_type> data(size);

	
	if (fixed) {
		static std::default_random_engine generator;
		std::generate(data.begin(), data.end(), []() { return distribution(generator); });
	}
	else {
		static std::default_random_engine generator{ std::random_device{}() };
		std::generate(data.begin(), data.end(), []() { return distribution(generator); });
	}

	return data;
}

void boardinit()
{

	for (int i = 0; i < BOARD_SIZE*BOARD_SIZE; i++) {
		board[i] = 0;
	}

	int maxoff = MINE_COUNT + FLAG_COUNT;


	std::vector<int> rv_x = generate_data(2 * maxoff, 0, BOARD_SIZE - 1);
	std::vector<int> rv_y = generate_data(2 * maxoff, 0, BOARD_SIZE - 1);

	//setType(BOARD_SIZE-1, BOARD_SIZE-1, FLAG);

	int i = 0;
	int c = 0;
	do	{
		int x = rv_x[i];
		int y = rv_y[i];

		//if (x == 0 && y == 0)
		//	continue;
		if (!isMine(x, y))
		{
			setType(x, y, MINE);
			c++;
		}
		
		i++;
	} while (c < MINE_COUNT);
	
	// set flag
	bool deploy_success = false;

	int f = 0;
	do {
		int x = rv_x[i];
		int y = rv_y[i];

		//if (x == 0 && y == 0)
		//	continue;
		if (!isMine(x, y)) {
			setType(x, y, FLAG);
			f++;
			if (f == FLAG_COUNT)
			{
				deploy_success = true;
				break;
			}
			
		}
	
		i++;
	} while (f < FLAG_COUNT &&i < MINE_COUNT + maxoff);

	if (!deploy_success)
		printf("deploy failed\n");

	glClearColor(0.8f, 0.8f, 0.8f, 1.f);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, window_width, 0, window_height, -1.f, 1.f);
	glPointSize(5.0);
	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	glEnable(GL_POINT_SMOOTH);
	glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}