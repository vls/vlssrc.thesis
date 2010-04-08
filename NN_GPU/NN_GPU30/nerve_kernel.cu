/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/* BP nerve neworl research
 * Device code.
 */

#ifndef _NERVE_KERNEL_H_
#define _NERVE_KERNEL_H_

#include <stdio.h>
// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define BLOCK_SIZE 16

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
////////////////////////////////////////////////////////////////////////////////
__global__ void
logsig1( float* A, int wA, int hA)
{
    const int posy = blockIdx.y * blockDim.y + threadIdx.y;
	const int posx = blockIdx.x * blockDim.x + threadIdx.x;
	const int pos = posy * wA + posx;

	if(posx < wA )
	{
	if(posy < hA )
    A[pos] = 1/(1+expf(-A[pos]));
	else if(posy == hA)
	A[pos] = 1.0f;
	}
}

__global__ void
logsig2( float* A, int wA, int hA)
{
    const int posy = blockIdx.y * blockDim.y + threadIdx.y;
	const int posx = blockIdx.x * blockDim.x + threadIdx.x;
	const int pos = posy * wA + posx;

	if((posx < wA)&&(posy < hA))
	{
    A[pos] = 1/(1+expf(-A[pos]));
	}
}

__global__ void
dotsub( float* C, float* A, float* B, int wA, int hA)
{
    const int posy = blockIdx.y * blockDim.y + threadIdx.y;
	const int posx = blockIdx.x * blockDim.x + threadIdx.x;
	const int pos = posy * wA + posx;

	if((posx < wA)&&(posy < hA))
	{
    C[pos] = A[pos]-B[pos];
	}
}

__global__ void
getError(float* diffMat, int w, int h)
{

}

__global__ void
getdelta( float* C, float* A, int wA, int hA)
{
    //const int tidx = threadIdx.x;
	//const int tidy = threadIdx.y;
	const int posy = blockIdx.y * blockDim.y + threadIdx.y;
	const int posx = blockIdx.x * blockDim.x + threadIdx.x;
	const int pos = posy * wA + posx;

	//__shared__ s_A[BLOCK_SIZE][BLOCK_SIZE];

	if((posx < wA)&&(posy < hA))
	{
	//s_A[tidy][tidx] = A[pos];
    C[pos] *= ((1.0f - A[pos])*A[pos]);
	}
	__syncthreads();
}

#endif
