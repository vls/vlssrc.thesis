#ifndef GLOBAL_H_INCLUDED
#define GLOBAL_H_INCLUDED

#include <boost/multi_array.hpp>
#include <boost/array.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include "windows.h"

typedef boost::multi_array<float, 2> Array2Df;

typedef float VALTYPE;


const int ROW = 21;
const int COL = 21;

const int OUTPUT = 10;

const float LEARNCOST = 0.5f;

const int NUM_LAYERS = 3;

static int UNITS[NUM_LAYERS] = {ROW*COL, 10, 4};



#define TIMEV_START(timeVar) {LARGE_INTEGER timeStart; LARGE_INTEGER freq; QueryPerformanceFrequency(&freq); timeVar *= freq.QuadPart; if (QueryPerformanceCounter(&timeStart));timeVar -= timeStart.QuadPart;}
#define TIMEV_END(timeVar) {LARGE_INTEGER timeStart; LARGE_INTEGER freq; QueryPerformanceFrequency(&freq); QueryPerformanceCounter(&timeStart);timeVar += timeStart.QuadPart; timeVar /= freq.QuadPart;}

#endif // GLOBAL_H_INCLUDED
