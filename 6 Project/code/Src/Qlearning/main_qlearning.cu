#include <stdlib.h>
#include <stdio.h>
#include <string.h>


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

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
//#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop
#include "draw_env.h"
#include "qlearning.h"
////////////////////////////////////////////////////////////////////////////////
#define REFRESH_DELAY     10 //ms
#define NUMBYTE4NODE      16 // xyz(12B)+rgba(4B)
extern unsigned int window_width;
extern unsigned int window_height;
// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;
StopWatchInterface *timer = NULL;

int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;

float3 camera_pos;
float3 camera_rot = { 0.0f,0.0f,0.0f };
unsigned int num_nodes;
bool KEY_CODES[256];
const float KEY_SPEED = 1.0f;
float prevtime;
float curtime;
float deltatime;
////////////////////////////////////////////////////////////////////////////////
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int steps = 0;
unsigned int episode = 0;
unsigned int active_agent = 0;
unsigned int flag_agent = 0;
int new_episode;
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char**argv);
bool initGLdevice(int argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags);
void runCuda(struct cudaGraphicsResource **vbo_resource);
void keyevent();
void draw_axis();
// callback functions
void display();
void keyboardDown(unsigned char key, int x, int y);
void keyboardUp(unsigned char key, int x, int y);
void mouseclick(int button, int state, int x, int y);
void mousemove(int x, int y);
void timerEvent(int value);
void cleanup();
////////////////////////////////////////////////////////////////////////////////
qlearningCls qlearning;
void init_visualization();
void draw();
void boardinit();
extern class Clock game_clock;
extern int board[];
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	num_nodes = 1000;
	// initialize opengl
	initGLdevice(argc, argv);
	init_visualization();
	// start rendering mainloop
	glutMainLoop();
	return 0;
}
////////////////////////////////////////////////////////////////////////////////
void init_visualization()
{
	//visrnd.init();
	//visrnd.readrate("acc_rate.bin");
	//visrnd.init_randomposition();
}
////////////////////////////////////////////////////////////////////////////////
bool initGLdevice(int argc, char **argv)
{
	sdkCreateTimer(&timer);

	if (false == initGL(&argc, argv)){
		return false;
	}

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	/*if (checkCmdLineFlag(argc, (const char **)argv, "device")) {
		if (gpuGLDeviceInit(argc, (const char **)argv) == -1) {
			return false;
		}
	} else	{
		cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
	}*/

#if defined (__APPLE__) || defined(MACOSX)
	atexit(cleanup);
#else
	glutCloseFunc(cleanup);
#endif

	// create VBO
	createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

	return true;
}
////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource)
{
	// map OpenGL buffer object for writing from CUDA
	float4 *dptr;
	checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
		*vbo_resource));

	new_episode = qlearning.learning(board, episode, steps);

	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
	glutInit(argc, argv);
	glutInitWindowPosition(0, 0);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_MULTISAMPLE);

	// create windows
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Final_project: Qlearning");

	// resigster callback functions
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboardDown);
	glutKeyboardUpFunc(keyboardUp);
	glutMouseFunc(mouseclick);
	glutMotionFunc(mousemove);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

	// initialize necessary OpenGL extensions
	if (!isGLVersionSupported(2, 0))
	{
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	boardinit();
	qlearning.init(BOARD_SIZE, board);

	SDK_CHECK_ERROR_GL();

	return true;
}

void computeFPS()
{
	char fps[512];
	
	sprintf(fps, "Final_project: Qlearning: episode=%d (steps=%d), TA=%d (FA=%3.1f%%, AA=%3.1f%%)", episode, steps, 
		NUM_AGENT, 100.0f*(float)flag_agent/NUM_AGENT, 100.0f*(float)active_agent / NUM_AGENT);

	glutSetWindowTitle(fps);
}
////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags)
{
	assert(vbo);

	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	unsigned int size = NUMBYTE4NODE*num_nodes;
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

	SDK_CHECK_ERROR_GL();
}
////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

	// unregister this buffer object with CUDA
	checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}
////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
	sdkStartTimer(&timer);
	curtime = ((float)glutGet(GLUT_ELAPSED_TIME)) / 1000;
	deltatime = curtime - prevtime;
	prevtime = curtime;

	keyevent();
	// run CUDA kernel to generate vertex positions
	runCuda(&cuda_vbo_resource);

	glClear(GL_COLOR_BUFFER_BIT);

	draw();

	glutSwapBuffers();
	glutPostRedisplay();

	sdkStopTimer(&timer);
	computeFPS();
}
////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboardDown(unsigned char key, int x, int y) {
	KEY_CODES[key] = true;
}

void keyboardUp(unsigned char key, int x, int y) {
	KEY_CODES[key] = false;
}

void keyevent()
{
	float step = KEY_SPEED*deltatime;

	if (KEY_CODES['a']) {
		camera_pos.x -= step;
	}
	if (KEY_CODES['d']) {
		camera_pos.x += step;
	}
	if (KEY_CODES['w']) {
		camera_pos.y += step;
	}
	if (KEY_CODES['x']) {
		camera_pos.y -= step;
	}
	if (KEY_CODES['s']) {
		camera_pos.z += step;
	}
	if (KEY_CODES['z']) {
		camera_pos.z -= step;
	}
	if (KEY_CODES['q']) {
#if defined(__APPLE__) || defined(MACOSX)
		exit(EXIT_SUCCESS);
#else
		glutDestroyWindow(glutGetWindow());
#endif
	}
}
////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouseclick(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1 << button;
	}
	else if (state == GLUT_UP)
	{
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void mousemove(int x, int y)
{
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1)
	{
		camera_rot.x += dy * 0.2f;
		camera_rot.y += dx * 0.2f;
	}
	else if (mouse_buttons & 4)
	{
		camera_pos.z += dy * 0.01f;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void timerEvent(int value)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	}
}

void cleanup()
{
	sdkDeleteTimer(&timer);

	if (vbo)
	{
		deleteVBO(&vbo, cuda_vbo_resource);
	}
}

void draw_axis()
{
	glBegin(GL_LINES);
	// x (red)
	glColor3f(1.0, 0.0, 0.0);
	glVertex3f(0, 0, 0);
	glVertex3f(0.1, 0, 0);
	glEnd();

	glBegin(GL_LINES);
	// y (green)
	glColor3f(0.0, 1.0, 0.0);
	glVertex3f(0, 0, 0);
	glVertex3f(0, 0.1, 0);
	glEnd();

	glBegin(GL_LINES);
	// z (blue)
	glColor3f(0.0, 0.0, 1.0);
	glVertex3f(0, 0, 0);
	glVertex3f(0, 0, 0.1);
	glEnd();
}

// drawings

