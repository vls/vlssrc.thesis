#include <stdio.h>
#include "Sample.h"
#include "nerve.h"
#include "Reader.h"
#include "test.h"
#include <cutil.h>
#include <cublas.h>
#include <cuda_runtime.h>


const int TRAINNUM = 20;




int
main( int argc,char** argv)
{
	printf("hello world\n");

	if (!InitCUDA())
	{
		return 0;
	}
	int iter = 1000;
	int trainnum = 20;
	bool isProfiler = false;
	int intProfiler = 0;
	cutGetCmdLineArgumenti(argc, (const char**) argv, "train", &trainnum);
	cutGetCmdLineArgumenti(argc, (const char**) argv, "iter", &iter);
	cutGetCmdLineArgumenti(argc, (const char**) argv, "profiler", &intProfiler);
	if(!intProfiler)
	{
		isProfiler = true;
	}

	printf("Iter = %d\n", iter);
	printf("TrainNum = %d\n", trainnum);

	CUT_DEVICE_INIT(argc, argv);


	cublasStatus status;
	status = cublasInit();
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		printf("Can't init cublas\n");
		printf("%s\n", cudaGetErrorString(cudaGetLastError()));
		return -1;
	}


	Image* imageList = new Image[trainnum];
	read64("my_optdigits.tra", imageList, trainnum);

	const int warmUpTime = 3;
	if(!isProfiler)
	{
		for(int i=0;i< warmUpTime;i++)
		{
			runImage(argc, argv, imageList, trainnum < warmUpTime ? trainnum : warmUpTime, 10, true);
		}
		printf("warmUp complete.\n\n\n");
	}
	
	runImage(argc, argv, imageList, trainnum, iter, true);
	delete[] imageList;
	//TestReduce();
	cublasShutdown();
	if(!isProfiler)
	{
		CUT_EXIT(argc, argv);
	}
	
	//getchar();
	return 0;
}