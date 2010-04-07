#include <stdio.h>
#include <cuda_runtime.h>
#include <cutil.h>


#if __DEVICE_EMULATION__

bool InitCUDA(void){return true;}

#else
bool InitCUDA(void)
{
	int count = 0;
	int i = 0;

	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	for(i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if(prop.major >= 1) {
				break;
			}
		}
	}
	if(i == count) {
		fprintf(stderr, "There is no device supporting CUDA.\n");
		return false;
	}
	cudaSetDevice(i);

	printf("CUDA initialized.\n");
	
	int driverVersion, runtimeVersion;
	cudaDriverGetVersion(&driverVersion);
	printf("  CUDA Driver Version:                           %d.%d\n", driverVersion/1000, driverVersion%100);
	cudaRuntimeGetVersion(&runtimeVersion);
	printf("  CUDA Runtime Version:                          %d.%d\n", runtimeVersion/1000, runtimeVersion%100);
	return true;
}

#endif
