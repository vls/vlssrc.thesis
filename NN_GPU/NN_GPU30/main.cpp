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
	int testnum = -1;
	float maxtime = 0.0f;
	cutGetCmdLineArgumenti(argc, (const char**) argv, "train", &trainnum);
	cutGetCmdLineArgumenti(argc, (const char**) argv, "iter", &iter);
	cutGetCmdLineArgumenti(argc, (const char**) argv, "profiler", &intProfiler);
	cutGetCmdLineArgumenti(argc, (const char**) argv, "test", &testnum);
	cutGetCmdLineArgumentf(argc, (const char**) argv, "maxtime", &maxtime);
	printf("%d\n", intProfiler);
	if(intProfiler)
	{
		isProfiler = true;
	}
	if(testnum == -1) testnum = trainnum /2;
	printf("Iter = %d\n", iter);
	printf("TrainNum = %d\n", trainnum);
	printf("TestNum = %d\n", testnum);

	CUT_DEVICE_INIT(argc, argv);


	cublasStatus status;
	status = cublasInit();
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		printf("Can't init cublas\n");
		printf("%s\n", cudaGetErrorString(cudaGetLastError()));
		return -1;
	}


	Image* imageList = new Image[trainnum+testnum];
	read64("my_optdigits.tra", imageList, trainnum + testnum);

	const int warmUpTime = 3;
	if(!isProfiler)
	{
		freopen("verbose.txt", "w", stdout);
		for(int i=0;i< warmUpTime;i++)
		{
			runImage(argc, argv, imageList, trainnum < warmUpTime ? trainnum : warmUpTime, 0, 10, false, 0.0f);
		}
		freopen("CON", "w", stdout);
		printf("Warm-up complete.\n\n\n");
	}
#ifdef _DEBUG
	freopen("out.txt", "w", stdout);
#endif // _DEBUG
	runImage(argc, argv, imageList, trainnum, testnum, iter, true, maxtime);
	freopen("CON", "w", stdout);
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