#include "MaNeural.h"

#include <iostream>


using namespace std;


MaNeural::MaNeural(int* units, float learnRate, TargetGenBase* tarptr)
{
	this->units = units;
	
	this->alpha = 0.9;
	this->eta = learnRate;

	this->tarptr = tarptr;
}

void MaNeural::TrainSet(Image* imageList, int count, float diff, int maxIter, float nStep, float maxStep)
{
	nStep = this->eta;

	

	for(int i=0; i < count; i++)
	{
		int length = imageList[i].length;
		for(int j=0;j<length;j++)
		{
			this->mInputValue.m_pTMatrix (j, i) = imageList[i].content[j] == 0 ? LOW : HIGH;
			
		}
		

	}
	//cout << "First mInputValue" << endl;
	//this->mInputValue.Print();
	Forward(this->mI2HWeight, this->mHideBias, this->mH2OWeight, this->mOutputBias);

	CMatrix mDemoOutput(this->numOutput, this->numSample);

	for(int i=0;i < count;i++)
	{
		float* target = tarptr->GetTarget(imageList[i].label);
		for(int j = 0;j<this->numOutput;j++)
		{
			mDemoOutput.m_pTMatrix (j, i) = target[j];
		}
	}

	CMatrix mOutputError(this->numOutput, this->numSample);
	mOutputError = mDemoOutput - this->mOutOutput;

	//cout << "mDemoOutput "<< endl;
	//mDemoOutput.Print();
	//cout << "mOutputError" << endl;
	//mOutputError.Print();


	float sysErrOld = mOutputError.GetSystemError();

	for(int nLoopTimes=1; nLoopTimes < maxIter; nLoopTimes++)	
	{
		//printf("loop = %d\n", nLoopTimes);
		if(sysErrOld < diff)
		{
			nLoopTimes--;
			break;
		}

		// 求输出层的delta值
		// 注意: 此处'/' 是 '点乘'!!!
		CMatrix	mOutDelta (this->numOutput, this->numSample);
		mOutDelta = (this->mOutOutput - this->mOutOutput  / this->mOutOutput ) / mOutputError;
		//cMatrixOutputLayerDelta 是一个误差矩阵，行数为输出元数，列数为样本数

		CMatrix t_mH2OWeight (this->mH2OWeight.GetColCount() , this->mH2OWeight.GetRowCount());
		t_mH2OWeight = this->mH2OWeight.Transpose();

		// 求隐含层的delta值
		// 注意: 此处'/' 是 '点乘'!!!
		CMatrix mHideDelta;
		mHideDelta.CopyMatrix ( (this->mHideOutput - (this->mHideOutput / this->mHideOutput)) / ( t_mH2OWeight * mOutDelta) );
		
		//cout << "mHideDelta" << endl;
		//mHideDelta.Print();

		// 定义新的输入层到隐含层的权值
		CMatrix mNewI2HWeight (this->mI2HWeight.GetRowCount(), this->mI2HWeight.GetColCount());
		// 定义的新的隐含层的阀值
		CMatrix mNewHideBias (this->numHidden, this->numSample);
		// 定义新的隐含层到输出层的权值
		CMatrix mNewH2OWeight (this->mH2OWeight.GetRowCount(), this->mH2OWeight.GetColCount());
		// 定义新的输出层的阀值
		CMatrix mNewOutputBias (this->numOutput, this->numSample);
		// 定义新的误差矩阵
		CMatrix cMatrixNewOutputLayerError(this->numOutput, this->numSample);


		// 权值和阀值调整
		mNewH2OWeight = mOutDelta * (this->mHideOutput.Transpose ()) * (nStep);
		mNewOutputBias = mOutDelta * nStep;//这里阀值的改变没有*nStep

		mNewI2HWeight = mHideDelta * (this->mInputValue.Transpose ()) * (nStep);
		//cout << "mInputValue" << endl;
		//this->mInputValue.Print();

		mNewHideBias = mHideDelta * nStep;

		// 赋值
		CMatrix tempI2HWeight = this->mI2HWeight;

		tempI2HWeight += mNewI2HWeight;
		//this->mI2HWeight += mNewI2HWeight;

		//cout << "new I2H weight" << endl;

		//mNewI2HWeight.Print();

		CMatrix tempHideBias(this->numHidden, 1);
		
		mNewHideBias.CopySubMatrix(tempHideBias, 0, mNewHideBias.GetColCount() - 1);
		tempHideBias += this->mHideBias;

		CMatrix tempH2OWeight = this->mH2OWeight;
		tempH2OWeight += mNewH2OWeight;

		CMatrix tempOutputBias(this->numOutput, 1);
		mNewOutputBias.CopySubMatrix(tempOutputBias, 0, mNewOutputBias.GetColCount() - 1);
		tempOutputBias += this->mOutputBias;

		// 前向计算
		Forward(tempI2HWeight, tempHideBias, tempH2OWeight, tempOutputBias);


		cMatrixNewOutputLayerError = mDemoOutput - this->mOutOutput;;
		float sysErrNew =	cMatrixNewOutputLayerError.GetSystemError ();

		mOutputError = cMatrixNewOutputLayerError;

		if(sysErrNew < sysErrOld)
		{
			this->mI2HWeight = tempI2HWeight;
			this->mH2OWeight = tempH2OWeight;
			this->mHideBias = tempHideBias;
			this->mOutputBias = tempOutputBias;
			sysErrOld = sysErrNew;
			nStep = nStep * 10 <= eta ? nStep * 10 : nStep;
		}
		else
		{
			
			nStep = nStep * -0.1 > 0.00001 ? nStep * -0.1 : nStep;//误差增大了，向反方向，走慢点
		}

		
		printf("loop = %d, New = %.6f, Old = %.6f, nStep = %.5f\n", nLoopTimes, sysErrNew, sysErrOld, nStep);



	}
}

bool MaNeural::Test(float* input, int label)
{
	return false;
}

void MaNeural::TestSet(Image* imageList, int count)
{
}