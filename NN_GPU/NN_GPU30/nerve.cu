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
* Host code.
*/
#include "nerve.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>
#include <cublas.h>
#include <nerve_kernel.h>
#include "Image.h"
#include "Reduce.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

////////////////////////////////////////////////////////////////////////////////
// declaration, forward




////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
const int OUTPUT_NUM = 4;

const int TARGET_NUM = 10;
float target[TARGET_NUM][OUTPUT_NUM];

const float HIGH = 1.0f;
const float LOW = 0.0f;

int
run(int argc, char** argv)
{	
	const float lr = 0.1f;
	const int MaxEpochs = 1;
	const int HiddenUnitNum = 10;
	const int InDim = 2;
	const int OutDim = 3;
	const int SamNum = 200;

	const int size_SamInEx = ((int)(SamNum*(InDim+1)));
	const int size_SamOut =  ((int)(SamNum*OutDim));
	const int size_W1Ex	 = ((int)(HiddenUnitNum*(InDim+1)));
	const int size_W2Ex	 = ((int)(OutDim*(HiddenUnitNum+1)));

	float* h_SamInEx;
	float* h_SamOut;
	float* h_W1Ex;
	float* h_W2Ex;
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	CUT_DEVICE_INIT(argc, argv);
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));


	cublasStatus status;
	status = cublasInit();
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		printf("Can't init cublas\n");
		printf("%s\n", cudaGetErrorString(cudaGetLastError()));
		return -1;
	}

	printf("%s\n", cudaGetErrorString(cudaGetLastError()));


	CUDA_SAFE_CALL( cudaMallocHost((void**) &h_SamInEx, sizeof(float)*size_SamInEx));
	CUDA_SAFE_CALL( cudaMallocHost((void**) &h_SamOut, sizeof(float)*size_SamOut));
	CUDA_SAFE_CALL( cudaMallocHost((void**) &h_W1Ex, sizeof(float)*size_W1Ex));
	CUDA_SAFE_CALL( cudaMallocHost((void**) &h_W2Ex, sizeof(float)*size_W2Ex));

	if(!InitSample(SamNum, InDim, OutDim, HiddenUnitNum, h_SamInEx, h_SamOut, h_W1Ex, h_W2Ex))
	{
		printf("Can't init sample\n");
		return 0;
	}
	train(lr, MaxEpochs, HiddenUnitNum, InDim, OutDim, SamNum, h_SamInEx, h_SamOut, h_W1Ex, h_W2Ex, false, 0.0f);
	/*
	printf("\n");
	for(int i=0; i<size_W1Ex; i++)
	{
	printf("%3.3f ",h_W1Ex[i]);
	}

	printf("\n");
	for(int i=0; i<size_W2Ex; i++)
	{
	printf("%3.3f ",h_W2Ex[i]);
	}
	*/
	FILE *p;
	p = fopen("nervedata0_W1Ex.dat", "wb");
	fwrite(h_W1Ex,sizeof(float),size_W1Ex,p);
	fclose(p);

	p = fopen("nervedata0_W2Ex.dat", "wb");
	fwrite(h_W2Ex,sizeof(float),size_W2Ex,p);
	fclose(p);


	CUDA_SAFE_CALL( cudaFreeHost((h_SamInEx)));
	CUDA_SAFE_CALL( cudaFreeHost((h_SamOut)));
	CUDA_SAFE_CALL( cudaFreeHost((h_W1Ex)));
	CUDA_SAFE_CALL( cudaFreeHost((h_W2Ex)));

	h_SamInEx = NULL;
	h_SamOut = NULL;
	h_W1Ex=NULL;
	h_W2Ex=NULL;

	cublasShutdown();
	CUT_EXIT(argc, argv);

	printf("run complete\n");
}


int runImage(int argc, char** argv, Image* imageList, int trainnum, int testnum, int maxIter, bool changeEta, float maxtime)
{
	if(imageList == NULL || trainnum == 0)
	{
		return -1;
	}
	
	const float lr = 0.1f;
	const int MaxEpochs = maxIter;
	const int HiddenUnitNum = 16;
	const int InDim = imageList[0].length;
	const int OutDim = OUTPUT_NUM;
	int SamNum = trainnum;

	int size_SamInEx = ((int)(SamNum*(InDim+1)));
	int size_SamOut =  ((int)(SamNum*OutDim));
	const int size_W1Ex	 = ((int)(HiddenUnitNum*(InDim+1)));
	const int size_W2Ex	 = ((int)(OutDim*(HiddenUnitNum+1)));

	float* h_SamInEx;
	float* h_SamOut;
	float* h_W1Ex;
	float* h_W2Ex;
	
	CUDA_SAFE_CALL( cudaMallocHost((void**) &h_SamInEx, sizeof(float)*size_SamInEx));
	CUDA_SAFE_CALL( cudaMallocHost((void**) &h_SamOut, sizeof(float)*size_SamOut));
	CUDA_SAFE_CALL( cudaMallocHost((void**) &h_W1Ex, sizeof(float)*size_W1Ex));
	CUDA_SAFE_CALL( cudaMallocHost((void**) &h_W2Ex, sizeof(float)*size_W2Ex));
	
	if(!InitImage(SamNum, InDim, OutDim, HiddenUnitNum, h_SamInEx, h_SamOut, h_W1Ex, h_W2Ex, imageList))
	{
		printf("Can't init image into input\n");
		return 0;
	}
	train(lr, MaxEpochs, HiddenUnitNum, InDim, OutDim, SamNum, h_SamInEx, h_SamOut, h_W1Ex, h_W2Ex, changeEta, maxtime);

	if(testnum != 0)
	{
		if(testnum != trainnum)
		{
			SamNum = testnum;
			size_SamInEx = ((int)(SamNum*(InDim+1)));
			size_SamOut =  ((int)(SamNum*OutDim));
			CUDA_SAFE_CALL( cudaFreeHost((h_SamOut)));
			CUDA_SAFE_CALL( cudaFreeHost((h_SamInEx)));
			CUDA_SAFE_CALL( cudaMallocHost((void**) &h_SamInEx, sizeof(float)*size_SamInEx));
			CUDA_SAFE_CALL( cudaMallocHost((void**) &h_SamOut, sizeof(float)*size_SamOut));
		}
		
		Image* testEntrance = imageList+trainnum;
		for(int i=0; i<SamNum; i++)
		{	
			for(int j=0; j<InDim;j++)
			{
				h_SamInEx[IDX2C(i,j,SamNum)] = testEntrance[i].content[j] / 16;
			}
		}
		Test(h_SamInEx, InDim, HiddenUnitNum, OutDim, SamNum, h_W1Ex, h_W2Ex, testEntrance);
	}
	

	CUDA_SAFE_CALL( cudaFreeHost((h_SamInEx)));
	CUDA_SAFE_CALL( cudaFreeHost((h_SamOut)));
	CUDA_SAFE_CALL( cudaFreeHost((h_W1Ex)));
	CUDA_SAFE_CALL( cudaFreeHost((h_W2Ex)));

	h_SamInEx = NULL;
	h_SamOut = NULL;
	h_W1Ex=NULL;
	h_W2Ex=NULL;

	

	printf("run complete\n");
}

int 
iDivUp(int a, int b){
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}


bool InitImage(int SamNum, int InDim, int OutDim, int HiddenUnitNum, float* h_SamInEx, float* h_SamOut, float* h_W1Ex,float* h_W2Ex, Image* imageList)
{
	const int size_SamInEx = ((int)(SamNum*(InDim+1)));
	const int size_SamOut =  ((int)(SamNum*OutDim));
	const int size_W1Ex	 = ((int)((InDim+1)*HiddenUnitNum));
	const int size_W2Ex	 = ((int)((HiddenUnitNum+1)*OutDim));
	


	for(int i=0; i<size_SamInEx; i++)
	{
		h_SamInEx[i] = 1.0f;
	}

	for(int i=0; i<SamNum; i++)
	{	
		for(int j=0; j<InDim;j++)
		{
			h_SamInEx[IDX2C(i,j,SamNum)] = imageList[i].content[j] / 16;
		}
	}
	/*
	#ifdef _DEBUG
	printf("Input:\n");
	for(int i=0; i<SamNum; i++)
	{	
	for(int j=0; j<InDim;j++)
	{
	printf("%.6f\t", h_SamInEx[IDX2C(i,j,SamNum)]);
	}
	printf("\n");
	}
	#endif // _DEBUG
	*/

	
	for(int i=0;i<TARGET_NUM;i++)
	{
		int fi = i;
		int j = OUTPUT_NUM -1;
		memset(target[i], 0, sizeof(float) * OUTPUT_NUM);
		while(fi)
		{
			target[i][j--] =(float)( fi & 1 ? HIGH : LOW);
			fi >>= 1;
		}
	}

	for(int i=0; i<SamNum; i++)
	{	
		for(int j=0; j<OutDim;j++)
		{

			h_SamOut[IDX2C(i,j,SamNum)] = target[imageList[i].label][j];
			//printf("%.2f\t", h_SamOut[IDX2C(i,j,SamNum)]);
		}
		//printf("\n");
	}
	/*
	#ifdef _DEBUG
	printf("Output:\n");
	for(int i=0; i<SamNum; i++)
	{	
	for(int j=0; j<OutDim;j++)
	{

	printf("%.3f\t",h_SamOut[IDX2C(i,j,SamNum)]);
	}
	printf("\n");
	}
	#endif // _DEBUG
	*/
	

	FILE* fptr = fopen("w1.txt", "r");
	FILE* fptr2 = fopen("w2.txt", "r");
	if(fptr != NULL && fptr2 != NULL)
	{
		printf("Weight files exist, reading files...\n");
		for(int i=0; i<InDim+1; i++)
		{
			for(int j=0;j<HiddenUnitNum;j++)
			{
				fscanf(fptr, "%f", &h_W1Ex[IDX2C(i,j,InDim+1)]);
			}

		}
		

		for(int i=0; i<HiddenUnitNum+1; i++)
		{
			for(int j=0;j<OutDim;j++)
			{
				fscanf(fptr2, "%f", &h_W2Ex[IDX2C(i,j,HiddenUnitNum+1)]);
				
			}

		}

		fclose(fptr);
		fclose(fptr2);
	}
	else
	{
		srand(time(NULL));
		printf("Files not found completely, generating...\n");
		for(int i=0; i<size_W1Ex; i++)
		{
			h_W1Ex[i]=0.2f*rand()/(float)RAND_MAX - 0.1f;
		}
		for(int i=0; i<size_W2Ex; i++)
		{
			h_W2Ex[i]=0.2f*rand()/(float)RAND_MAX - 0.1f;
		}

		freopen("w1.txt", "w", stdout);
		for(int i=0; i<InDim+1; i++)
		{
			for(int j=0;j<HiddenUnitNum;j++)
			{
				printf("%f\t",h_W1Ex[IDX2C(i,j,InDim+1)]);
			}

		}

		freopen("w2.txt", "w", stdout);
		for(int i=0; i<HiddenUnitNum+1; i++)
		{
			for(int j=0;j<OutDim;j++)
			{
				printf("%f\t", h_W2Ex[IDX2C(i,j,HiddenUnitNum)]);
			}

		}

		freopen("CON", "w",stdout);
	}
	

	
	
	
	
	
	return true;
}

bool InitSample(int SamNum, int InDim, int OutDim, int HiddenUnitNum, float* h_SamInEx, float* h_SamOut, float* h_W1Ex,float* h_W2Ex)
{
	const int size_SamInEx = ((int)(SamNum*(InDim+1)));
	const int size_SamOut =  ((int)(SamNum*OutDim));
	const int size_W1Ex	 = ((int)(HiddenUnitNum*(InDim+1)));
	const int size_W2Ex	 = ((int)(OutDim*(HiddenUnitNum+1)));


	srand(clock());
	FILE *p;
	p = fopen("nervedata\\samin.dat", "rb");

	if(!p) return false;

	for(int i=0; i<size_SamInEx; i++)
	{
		h_SamInEx[i] = 1.0f;
	}

	for(int i=0; i<SamNum; i++)
	{	
		for(int j=0; j<InDim;j++)
		{
			fread(&h_SamInEx[j*SamNum+i],sizeof(float),1,p);
		}
	}
	fclose(p);

	for(int i=0; i<size_SamInEx; i++)
	{
		//printf("%3.4f\t",h_SamInEx[i]);
		//if((i + 1) % InDim == 0) printf("\n");
	}

	printf("\n");

	p = fopen("nervedata\\samout.dat", "rb");

	if(!p) return false;

	for(int i=0; i<SamNum; i++)
	{	
		for(int j=0; j<OutDim;j++)
		{
			fread(&h_SamOut[j*SamNum+i],sizeof(float),1,p);
		}
	}
	for(int i=0; i<size_SamOut; i++)
	{
		//printf("%3.5f\t",h_SamOut[i]);
		//if((i + 1) % OutDim == 0) printf("\n");
	}
	fclose(p);

	printf("\n");

	for(int i=0; i<size_W1Ex; i++)
	{
		h_W1Ex[i]=0.2f*rand()/(float)RAND_MAX - 0.1f;
	}
	for(int i=0; i<size_W2Ex; i++)
	{
		h_W2Ex[i]=0.2f*rand()/(float)RAND_MAX - 0.1f;
	}
	/*
	p = fopen("I:\\temp\\NN_GPU\\nervedataW1Ex.dat", "rb");
	for(int i=0; i<(InDim+1); i++)
	{	
	for(int j=0; j<HiddenUnitNum;j++)
	{
	fread(&h_W1Ex[j*(InDim+1)+i],sizeof(float),1,p);
	}
	}

	for(int i=0; i<size_W1Ex; i++)
	{
	printf("%3.3f ",h_W1Ex[i]);
	}
	fclose(p);

	printf("\n");

	p = fopen("I:\\temp\\NN_GPU\\nervedataW2Ex.dat", "rb");
	for(int i=0; i<(HiddenUnitNum+1); i++)
	{	
	for(int j=0; j<OutDim;j++)
	{
	fread(&h_W2Ex[j*(HiddenUnitNum+1)+i],sizeof(float),1,p);
	}
	}

	for(int i=0; i<size_W2Ex; i++)
	{
	printf("%3.3f ",h_W2Ex[i]);
	}
	fclose(p);
	*/
	printf("\n");

	p=NULL;

	return true;
}



void
train(float lr, int MaxEpochs, int HiddenUnitNum, int InDim,int OutDim,int SamNum, float* h_SamInEx, float* h_SamOut, float* h_W1Ex, float* h_W2Ex, bool changeEta, float maxtime)
{
	const int size_SamInEx = ((int)(SamNum*(InDim+1)));
	const int size_SamOut =  ((int)(SamNum*OutDim));
	const int size_W1Ex	 = ((int)(HiddenUnitNum*(InDim+1)));
	const int size_W2Ex	 = ((int)(OutDim*(HiddenUnitNum+1)));

	float* d_W1Ex;
	float* d_W2Ex;
	float* d_W2;
	float* d_SamInEx;
	float* d_SamOut;
	float* HiddenOutEx;
	float* NetworkOut;
	float* Delta1;
	float* Delta2;
	float alpha = 1.05;
	float beta = 0.7;

	float preCopyTime, runTime, postCopyTime;
	

	unsigned int timer = 0;
	unsigned int copyTimer = 0;
	unsigned int postCopyTimer = 0;

/*
	for(int i=0;i<SamNum;i++)
	{
		for(int j=0;j<InDim+1;j++)
		{
			printf("%.4f\t", h_SamInEx[IDX2C(i,j,SamNum)]);
		}
		printf("\n");
	}
*/
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 HiddenOutExgrid(iDivUp(SamNum, BLOCK_SIZE), iDivUp((HiddenUnitNum+1), BLOCK_SIZE));
	dim3 HiddenOutgrid(iDivUp(SamNum, BLOCK_SIZE), iDivUp((HiddenUnitNum), BLOCK_SIZE));
	dim3 SamOutgrid(iDivUp(SamNum, BLOCK_SIZE), iDivUp(OutDim, BLOCK_SIZE));

	dim3 OutErrGrid(iDivUp(OutDim, BLOCK_SIZE), iDivUp(OutDim, BLOCK_SIZE));

	float* h_HiddenOutEx;
	h_HiddenOutEx = (float*) malloc(sizeof(float)*(HiddenUnitNum+1)*SamNum);
	CUT_SAFE_CALL( cutCreateTimer( &copyTimer));
	CUT_SAFE_CALL( cutStartTimer( copyTimer));
	

	CUDA_SAFE_CALL(cudaMalloc((void**) &d_W1Ex, sizeof(float)*size_W1Ex));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_W2Ex, sizeof(float)*size_W2Ex));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_W2, sizeof(float)*OutDim*HiddenUnitNum));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_SamInEx, sizeof(float)*size_SamInEx));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_SamOut, sizeof(float)*size_SamOut));
	CUDA_SAFE_CALL(cudaMalloc((void**) &HiddenOutEx, sizeof(float)*(HiddenUnitNum+1)*SamNum));
	CUDA_SAFE_CALL(cudaMalloc((void**) &NetworkOut, sizeof(float)*size_SamOut));
	CUDA_SAFE_CALL(cudaMalloc((void**) &Delta1, sizeof(float)*HiddenUnitNum*SamNum));
	CUDA_SAFE_CALL(cudaMalloc((void**) &Delta2, sizeof(float)*size_SamOut));  
	CUDA_SAFE_CALL(cudaMemcpy(d_W1Ex, h_W1Ex, sizeof(float)*size_W1Ex,cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_W2Ex, h_W2Ex, sizeof(float)*size_W2Ex,cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_SamInEx, h_SamInEx, sizeof(float)*size_SamInEx,cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_SamOut, h_SamOut, sizeof(float)*size_SamOut,cudaMemcpyHostToDevice));
	
	
	CUT_SAFE_CALL( cutStopTimer( copyTimer));
	preCopyTime = cutGetTimerValue( copyTimer);
	printf( "Pre-copy time: %f (ms)\n", cutGetTimerValue( copyTimer));
	CUT_SAFE_CALL( cutDeleteTimer( copyTimer));
	
	CUT_SAFE_CALL( cutCreateTimer( &timer));
	CUT_SAFE_CALL( cutStartTimer( timer));
	float errOld;
	float errNew;
	int forwardCount = 0;

	forwardCount++;
	for(int i=0; i< OutDim; i++)
	{
		CUDA_SAFE_CALL(cudaMemcpy(d_W2 + HiddenUnitNum * i, d_W2Ex + (HiddenUnitNum + 1) * i, sizeof(float)*HiddenUnitNum,cudaMemcpyDeviceToDevice));
		//注意这里是column-majoring
	}
	cublasSgemm('n','n',SamNum, HiddenUnitNum, (InDim+1), 1.0f,  d_SamInEx, SamNum, d_W1Ex, (InDim+1), 0.0f, HiddenOutEx, SamNum );
	

	

	logsig1<<<HiddenOutExgrid, threads>>>(HiddenOutEx, SamNum, HiddenUnitNum);

	//logsig1后边界的阈值设为1
	cublasSgemm('n','n',SamNum, OutDim, (HiddenUnitNum+1), 1.0f,  HiddenOutEx, SamNum, d_W2Ex, (HiddenUnitNum+1), 0.0f, NetworkOut, SamNum );
	logsig2<<<SamOutgrid,threads>>>(NetworkOut, SamNum, OutDim);
	//logsig2不设边界阈值
	
	
	
	

	dotsub<<<SamOutgrid,threads>>>(Delta2, d_SamOut, NetworkOut, SamNum, OutDim);
	//#ifdef VERBOSE
	errOld = GPUPowSum(Delta2, OutDim* SamNum);
	for(int l=0; l< MaxEpochs; l++)
	{



		getdelta<<<SamOutgrid,threads>>>(Delta2, NetworkOut, SamNum, OutDim);
		
		cublasSgemm('t','n',(HiddenUnitNum+1), OutDim, SamNum, lr,  HiddenOutEx, SamNum, Delta2, SamNum, 1.0f, d_W2Ex, (HiddenUnitNum+1) );

		//实质上将W变化量加起来了
		cublasSgemm('n','t', SamNum, HiddenUnitNum, OutDim, 1.0f,  Delta2, SamNum, d_W2, HiddenUnitNum, 0.0f, Delta1, SamNum );
		
		getdelta<<<HiddenOutgrid, threads>>>(Delta1, HiddenOutEx, SamNum, HiddenUnitNum);
		
		
		cublasSgemm('t','n', (InDim+1), HiddenUnitNum, SamNum, lr, d_SamInEx, SamNum, Delta1, SamNum, 1.0f, d_W1Ex, (InDim+1));
		

		forwardCount++;
		for(int i=0; i< OutDim; i++)
		{
			CUDA_SAFE_CALL(cudaMemcpy(d_W2 + HiddenUnitNum * i, d_W2Ex + (HiddenUnitNum + 1) * i, sizeof(float)*HiddenUnitNum,cudaMemcpyDeviceToDevice));
			//注意这里是column-majoring
		}

		cublasSgemm('n','n',SamNum, HiddenUnitNum, (InDim+1), 1.0f,  d_SamInEx, SamNum, d_W1Ex, (InDim+1), 0.0f, HiddenOutEx, SamNum );
		//SamNum*(InDim+1) * (InDim+1)*HiddenUnitNUm
		
		#ifdef _DEBUG
		CUDA_SAFE_CALL(cudaMemcpy(h_HiddenOutEx, HiddenOutEx, sizeof(float)*(HiddenUnitNum+1)*SamNum, cudaMemcpyDeviceToHost));
		#endif // _DEBUG
		

		logsig1<<<HiddenOutExgrid, threads>>>(HiddenOutEx, SamNum, HiddenUnitNum);
		//logsig1后边界的阈值设为1
		cublasSgemm('n','n',SamNum, OutDim, (HiddenUnitNum+1), 1.0f,  HiddenOutEx, SamNum, d_W2Ex, (HiddenUnitNum+1), 0.0f, NetworkOut, SamNum );
		logsig2<<<SamOutgrid,threads>>>(NetworkOut, SamNum, OutDim);
		//logsig2不设边界阈值
		dotsub<<<SamOutgrid,threads>>>(Delta2, d_SamOut, NetworkOut, SamNum, OutDim);

		errNew = GPUPowSum(Delta2, OutDim* SamNum);
	
		#ifdef _DEBUG
			printf("Iter = %d, New = %.6f, Old = %.6f, Eta = %.3f\n", l, errNew, errOld, lr);
		#endif // _DEBUG
		if(errNew < errOld)
		{
			errOld = errNew;
			if(changeEta)
			{
				float newEta = lr * 1.05;
				lr = newEta < 0.9 ? newEta : lr ;
			}
		}
		else if(errNew > errOld * 1.04)
		{
			if(changeEta)
			{
				float newEta = lr* 0.7;
				lr = newEta > 0.01 ? newEta : lr;
			}
		}
	
		if(maxtime != 0)
		{
			cutStopTimer(timer);
			float nowtime =  cutGetTimerValue( timer);
			cutStartTimer(timer);
			if(nowtime > maxtime)
			{
				
				break;
			}
		}

		

	}
	CUT_SAFE_CALL( cutStopTimer( timer));
	runTime = cutGetTimerValue( timer);
	printf( "Processing time: %f (ms)\n", cutGetTimerValue( timer));
	CUT_SAFE_CALL( cutDeleteTimer( timer));
	printf("Err = %.6f\n",errNew);
	printf("Forward Count = %d\n", forwardCount);

	CUT_SAFE_CALL( cutCreateTimer( &postCopyTimer));
	CUT_SAFE_CALL( cutStartTimer( postCopyTimer));
	CUDA_SAFE_CALL(cudaMemcpy(h_W1Ex, d_W1Ex, sizeof(float)*size_W1Ex,cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(h_W2Ex, d_W2Ex, sizeof(float)*size_W2Ex,cudaMemcpyDeviceToHost));

	CUT_SAFE_CALL( cutStopTimer( postCopyTimer));
	postCopyTime = cutGetTimerValue( postCopyTimer);
	printf( "Post-copy time: %f (ms)\n", cutGetTimerValue( postCopyTimer));
	CUT_SAFE_CALL( cutDeleteTimer( postCopyTimer));


	printf("Copy time percent = %.4f %%\n", (preCopyTime + postCopyTime) * 100 / (preCopyTime + postCopyTime + runTime) );

	
	#ifdef _DEBUG
	for(int i=0;i<SamNum;i++)
	{
	for(int j = 0;j<HiddenUnitNum +1;j++)
	{
	printf("%.3f\t", h_HiddenOutEx[IDX2C(i,j,SamNum)]);
	}
	printf("\n");
	}
	#endif // _DEBUG
	

	CUDA_SAFE_CALL(cudaFree(d_SamInEx));
	CUDA_SAFE_CALL(cudaFree(d_SamOut));
	CUDA_SAFE_CALL(cudaFree(d_W1Ex));
	CUDA_SAFE_CALL(cudaFree(d_W2Ex));
	CUDA_SAFE_CALL(cudaFree(d_W2));

	CUDA_SAFE_CALL(cudaFree(HiddenOutEx));
	CUDA_SAFE_CALL(cudaFree(NetworkOut));
	CUDA_SAFE_CALL(cudaFree(Delta1));
	CUDA_SAFE_CALL(cudaFree(Delta2));

	free(h_HiddenOutEx);
}

float Test(float* h_SamInEx, int InDim, int HiddenUnitNum, int OutDim ,int SamNum, float* h_W1Ex, float* h_W2Ex, Image* imageList)
{
	printf("Begin to test...\n");
	const int size_SamInEx = ((int)(SamNum*(InDim+1)));
	const int size_SamOut =  ((int)(SamNum*OutDim));
	const int size_W1Ex	 = ((int)(HiddenUnitNum*(InDim+1)));
	const int size_W2Ex	 = ((int)(OutDim*(HiddenUnitNum+1)));

	float* d_W1Ex;
	float* d_W2Ex;
	float* d_W2;
	float* d_SamInEx;
	float* HiddenOutEx;
	float* NetworkOut;


	CUDA_SAFE_CALL(cudaMalloc((void**) &d_W1Ex, sizeof(float)*size_W1Ex));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_W2Ex, sizeof(float)*size_W2Ex));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_W2, sizeof(float)*OutDim*HiddenUnitNum));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_SamInEx, sizeof(float)*size_SamInEx));
	CUDA_SAFE_CALL(cudaMalloc((void**) &HiddenOutEx, sizeof(float)*(HiddenUnitNum+1)*SamNum));
	CUDA_SAFE_CALL(cudaMalloc((void**) &NetworkOut, sizeof(float)*size_SamOut));

	CUDA_SAFE_CALL(cudaMemcpy(d_W1Ex, h_W1Ex, sizeof(float)*size_W1Ex,cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_W2Ex, h_W2Ex, sizeof(float)*size_W2Ex,cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_SamInEx, h_SamInEx, sizeof(float)*size_SamInEx,cudaMemcpyHostToDevice));

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 HiddenOutExgrid(iDivUp(SamNum, BLOCK_SIZE), iDivUp((HiddenUnitNum+1), BLOCK_SIZE));
	dim3 HiddenOutgrid(iDivUp(SamNum, BLOCK_SIZE), iDivUp((HiddenUnitNum), BLOCK_SIZE));
	dim3 SamOutgrid(iDivUp(SamNum, BLOCK_SIZE), iDivUp(OutDim, BLOCK_SIZE));

	dim3 OutErrGrid(iDivUp(OutDim, BLOCK_SIZE), iDivUp(OutDim, BLOCK_SIZE));

/*
	for(int i=0;i<SamNum;i++)
	{
		for(int j=0;j<InDim+1;j++)
		{
			printf("%.4f\t", h_SamInEx[IDX2C(i,j,SamNum)]);
		}
		printf("\n");
	}
*/

	float errOld;
	for(int i=0; i< OutDim; i++)
	{
		CUDA_SAFE_CALL(cudaMemcpy(d_W2 + HiddenUnitNum * i, d_W2Ex + (HiddenUnitNum + 1) * i, sizeof(float)*HiddenUnitNum,cudaMemcpyDeviceToDevice));
		//注意这里是column-majoring
	}
	cublasSgemm('n','n',SamNum, HiddenUnitNum, (InDim+1), 1.0f,  d_SamInEx, SamNum, d_W1Ex, (InDim+1), 0.0f, HiddenOutEx, SamNum );

	logsig1<<<HiddenOutExgrid, threads>>>(HiddenOutEx, SamNum, HiddenUnitNum);
	//logsig1后边界的阈值设为1
	cublasSgemm('n','n',SamNum, OutDim, (HiddenUnitNum+1), 1.0f,  HiddenOutEx, SamNum, d_W2Ex, (HiddenUnitNum+1), 0.0f, NetworkOut, SamNum );
	logsig2<<<SamOutgrid,threads>>>(NetworkOut, SamNum, OutDim);



	float* h_Out;
	h_Out = (float*) malloc(sizeof(float) * size_SamOut);
	CUDA_SAFE_CALL(cudaMemcpy(h_Out, NetworkOut, sizeof(float)*size_SamOut, cudaMemcpyDeviceToHost));

	int rightCount = 0;

	float door = (HIGH - LOW) * 0.7 + LOW;
	for(int i=0;i<SamNum;i++)
	{
		int predict = 0;
		for(int j=0;j<OutDim;j++)
		{
			predict <<= 1;
			float tar = h_Out[IDX2C(i,j, SamNum)];
			if(tar > door)
			{
				predict |= 1;
			}
			
			printf("%.3f\t", tar);
		}
		printf("\n");
		printf("%d -> %d\n", imageList[i].label, predict );
		if(imageList[i].label == predict)
		{
			rightCount++;
		}
	}

	printf("Percent = %.2f %%\n", (float)rightCount * 100 / SamNum);
	
	free(h_Out);

	//logsig2不设边界阈值

	CUDA_SAFE_CALL(cudaFree(d_SamInEx));
	CUDA_SAFE_CALL(cudaFree(d_W1Ex));
	CUDA_SAFE_CALL(cudaFree(d_W2Ex));
	CUDA_SAFE_CALL(cudaFree(d_W2));

	CUDA_SAFE_CALL(cudaFree(HiddenOutEx));
	CUDA_SAFE_CALL(cudaFree(NetworkOut));

}

void Print(float* arr, int row, int col, const char* str)
{
	float* h_arr;

	h_arr = (float*) malloc(sizeof(float) * row * col);

	if(h_arr == NULL)
	{
		return;
	}

	CUDA_SAFE_CALL(cudaMemcpy(h_arr, arr, row*col*sizeof(float), cudaMemcpyDeviceToHost));

	PrintHost(h_arr, row, col, str);
	free(h_arr);
}

void PrintHost(float* arr, int row, int col, const char* str)
{
	if(str != NULL)
	{
		printf("%s\n", str);
	}

	for(int i=0;i<row;i++)
	{
		for(int j=0;j<col;j++)
		{
			printf("%f\t", arr[IDX2C(i,j,row)]);
		}
		printf("\n");
	}
}


