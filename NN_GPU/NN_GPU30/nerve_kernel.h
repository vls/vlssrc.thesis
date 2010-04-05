#include <stdio.h>
// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define BLOCK_SIZE 16

__global__ void
logsig1( float* A, int wA, int hA);

__global__ void
logsig2( float* A, int wA, int hA);

__global__ void
dotsub( float* C, float* A, float* B, int wA, int hA);

__global__ void
getdelta( float* C, float* A, int wA, int hA);