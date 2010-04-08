#include <cuda_runtime.h>
#include <cutil.h>

unsigned int nextPow2( unsigned int x ) {
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}


////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel 
// 6, we observe the maximum specified number of blocks, because each thread in 
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
	threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
	blocks = (n + (threads * 2 - 1)) / (threads * 2);
}




#ifdef __DEVICE_EMULATION__
#define EMUSYNC __syncthreads()
#else
#define EMUSYNC
#endif

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
	__device__ inline operator       T*()
	{
		extern __shared__ int __smem[];
		return (T*)__smem;
	}

	__device__ inline operator const T*() const
	{
		extern __shared__ int __smem[];
		return (T*)__smem;
	}
};


template <class T, unsigned int blockSize>
__global__ void
reduce5(T *g_idata, T *g_odata, unsigned int n)
{
	// now that we are using warp-synchronous programming (below)
	// we need to declare our shared memory volatile so that the compiler
	// doesn't reorder stores to it and induce incorrect behavior.
	volatile T *sdata = SharedMemory<T>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;

	sdata[tid] = (i < n) ? g_idata[i]*g_idata[i] : 0;
	if (i + blockSize < n) 
		sdata[tid] += g_idata[i+blockSize] * g_idata[i+blockSize];  

	__syncthreads();

	// do reduction in shared mem
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

#ifndef __DEVICE_EMULATION__
	if (tid < 32)
#endif
	{
		if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; EMUSYNC; }
		if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; EMUSYNC; }
		if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; EMUSYNC; }
		if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; EMUSYNC; }
		if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; EMUSYNC; }
		if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; EMUSYNC; }
	}

	// write result for this block to global mem 
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


float GPUPowSum(float* gpu_diff, int length)
{
	float sum = 0;
	int maxBlocks = 64;
	int maxThreads = 256;
	int numThreads = 0;
	int numBlocks = 0;
	getNumBlocksAndThreads(length, maxBlocks, maxThreads, numBlocks, numThreads);


	dim3 dimBlock(numThreads, 1, 1);
	dim3 dimGrid(numBlocks, 1, 1);
	int smemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(float) : numThreads * sizeof(float);


	float* d_odata;
	float* h_odata;
	h_odata = (float*) malloc(sizeof(float) * numBlocks);

	CUDA_SAFE_CALL(cudaMalloc((void**) &d_odata, sizeof(float)* numBlocks));


	switch (numThreads)
	{
	case 512:
		reduce5<float, 512><<< dimGrid, dimBlock, smemSize >>>(gpu_diff, d_odata, length); break;
	case 256:
		reduce5<float, 256><<< dimGrid, dimBlock, smemSize >>>(gpu_diff, d_odata, length); break;
	case 128:
		reduce5<float, 128><<< dimGrid, dimBlock, smemSize >>>(gpu_diff, d_odata, length); break;
	case 64:
		reduce5<float,  64><<< dimGrid, dimBlock, smemSize >>>(gpu_diff, d_odata, length); break;
	case 32:
		reduce5<float,  32><<< dimGrid, dimBlock, smemSize >>>(gpu_diff, d_odata, length); break;
	case 16:
		reduce5<float,  16><<< dimGrid, dimBlock, smemSize >>>(gpu_diff, d_odata, length); break;
	case  8:
		reduce5<float,   8><<< dimGrid, dimBlock, smemSize >>>(gpu_diff, d_odata, length); break;
	case  4:
		reduce5<float,   4><<< dimGrid, dimBlock, smemSize >>>(gpu_diff, d_odata, length); break;
	case  2:
		reduce5<float,   2><<< dimGrid, dimBlock, smemSize >>>(gpu_diff, d_odata, length); break;
	case  1:
		reduce5<float,   1><<< dimGrid, dimBlock, smemSize >>>(gpu_diff, d_odata, length); break;
	}

	CUDA_SAFE_CALL(cudaMemcpy(h_odata, d_odata, sizeof(float)* numBlocks, cudaMemcpyDeviceToHost));

	for(int i=0;i<numBlocks;i++)
	{
		//printf("block [%d] = %.6f\n", i, h_odata[i]);
		sum += h_odata[i];
	}


	CUDA_SAFE_CALL(cudaFree(d_odata));
	free(h_odata);

	return sum;

}

float HostPowSum(float* diff, int length)
{
	float sum = 0;

	float* d_diff;
	
	
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_diff, sizeof(float)* length));
	
	CUDA_SAFE_CALL(cudaMemcpy(d_diff, diff, length*sizeof(float), cudaMemcpyHostToDevice));

	sum = GPUPowSum(d_diff, length);
	
	CUDA_SAFE_CALL(cudaFree(d_diff));
	
	return sum;

}

