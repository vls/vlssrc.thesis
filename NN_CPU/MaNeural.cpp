#include "MaNeural.h"

#include <iostream>
const int _NUMLAYER = 3;
#define VERBOSE false
using namespace std;

int forwardCount = 0;
MaNeural::MaNeural(int* units, float eta, TargetGenBase* tarptr)
{
	this->units = units;
	
	this->alpha = 0.5;
	this->eta = eta;

	this->tarptr = tarptr;
}

MaNeural::~MaNeural()
{
	delete[] this->units;
}

void MaNeural::Init(int numSample)
{
	this->numInput = this->units[0];
	this->numOutput = this->units[_NUMLAYER-1];
	this->numHidden = this->units[1];

	this->numSample = numSample;

	this->mInputValue.Resize(numInput, numSample);

	this->mI2HWeight.Resize(numHidden, numInput);


	this->mH2OWeight.Resize(numOutput, numHidden);


	this->mHideBias.Resize(numHidden, 1);


	this->mOutputBias.Resize(numOutput, 1);

	for (int i = 0;i<mHideBias.GetRowCount();i++)
	{
		mHideBias.m_pTMatrix(i,0) = 1.0f;
	}

	for (int i = 0;i<mOutputBias.GetRowCount();i++)
	{
		mOutputBias.m_pTMatrix(i,0) = 1.0f;
	}

}

void MaNeural::GenerateWeight()
{
	this->mI2HWeight.RandomInitialize(HIGH, LOW);
	this->mH2OWeight.RandomInitialize(HIGH, LOW);
	
}

void MaNeural::TrainSet(float* input, float* target, int count, float diff, int maxIter, bool changeEta, bool deltaWeight)
{
	this->mInputValue.Resize(this->numInput, this->numSample);
	this->mDemoOutput.Resize(this->numOutput, this->numSample);

	for(int j=0;j<this->numSample;j++)
		for(int i=0;i<this->numInput;i++)
		{
			
			this->mInputValue.m_pTMatrix(i, j) = input[i + j*this->numInput];
		}
	for(int j=0;j<this->numSample;j++)
		for(int i=0;i<this->numOutput;i++)
		{
			this->mDemoOutput.m_pTMatrix(i, j) = target[i + j*this->numInput];
		}

	__TrainSet(count, diff, maxIter, changeEta, deltaWeight);
}

void MaNeural::__TrainSet(int count, float diff, int maxIter, bool changeEta, bool deltaWeight, float maxtime)
{
	forwardCount = 0;


	CMatrix mOutputError(this->numOutput, this->numSample);
	double t;
	TIMEV_START(t);


	Forward(this->mI2HWeight, this->mHideBias, this->mH2OWeight, this->mOutputBias, VERBOSE);
	mOutputError = mDemoOutput - this->mOutOutput;
	float sysErrOld = mOutputError.GetSystemError();

	this->s_mI2HWeight = this->mI2HWeight;
	this->s_mH2OWeight = this->mH2OWeight;

	if(VERBOSE)
	{
		cout << "mDemoOutput "<< endl;
		mDemoOutput.Print();
	}
	
	//cout << "mOutputError" << endl;
	//mOutputError.Print();

	float sysErrNew;
	
	int nLoopTimes;
	for(nLoopTimes=1; nLoopTimes < maxIter; nLoopTimes++)	
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



		// 定义新的输入层到隐含层的权值
		CMatrix mNewI2HWeight (this->mI2HWeight.GetRowCount(), this->mI2HWeight.GetColCount());

		/*
		// 定义的新的隐含层的阀值
		CMatrix mNewHideBias (this->numHidden, this->numSample);
		*/
		// 定义新的隐含层到输出层的权值
		CMatrix mNewH2OWeight (this->mH2OWeight.GetRowCount(), this->mH2OWeight.GetColCount());

		/*
		// 定义新的输出层的阀值
		CMatrix mNewOutputBias (this->numOutput, this->numSample);
		*/
		// 定义新的误差矩阵
		CMatrix cMatrixNewOutputLayerError(this->numOutput, this->numSample);


		// 权值和阀值调整
		mNewH2OWeight = mOutDelta * (this->mHideOutput.Transpose ()) * (this->eta);


		//mNewOutputBias = mOutDelta * this->eta;//这里阀值的改变没有*nStep

		mNewI2HWeight = mHideDelta * (this->mInputValue.Transpose ()) * (this->eta);



		//mNewHideBias = mHideDelta * this->eta;

		// 赋值
		this->mI2HWeight += mNewI2HWeight;
		if(deltaWeight)
		{
			if(d_mI2HWeight.GetColCount() == 0 || d_mI2HWeight.GetRowCount() == 0)
			{
				this->d_mI2HWeight = this->mI2HWeight;
			}
			else
			{
				this->mI2HWeight += this->d_mI2HWeight * alpha;
			}
			
		}


		this->mH2OWeight += mNewH2OWeight;
		if(deltaWeight)
		{
			if(d_mH2OWeight.GetColCount() == 0 || d_mH2OWeight.GetRowCount() == 0)
			{
				this->d_mH2OWeight = this->mH2OWeight;
			}
			else
			{
				this->mH2OWeight += this->d_mH2OWeight * alpha;
			}
		}


		/*
		CMatrix tempHideBias(this->numHidden, 1);
		mNewHideBias.CopySubMatrix(tempHideBias, 0, mNewHideBias.GetColCount() - 1);
		tempHideBias += this->mHideBias;

		CMatrix tempOutputBias(this->numOutput, 1);
		mNewOutputBias.CopySubMatrix(tempOutputBias, 0, mNewOutputBias.GetColCount() - 1);
		tempOutputBias += this->mOutputBias;
		*/




		//save delta weight
		if(deltaWeight)
		{
			this->d_mH2OWeight = mNewH2OWeight;
			this->d_mI2HWeight = mNewI2HWeight;
		}


		// 前向计算
		Forward(this->mI2HWeight, this->mHideBias, this->mH2OWeight, this->mOutputBias, VERBOSE);


		mOutputError = mDemoOutput - this->mOutOutput;;
		sysErrNew =	mOutputError.GetSystemError ();

		alpha = this->alpha;


		bool restore = false;
		if(sysErrNew < sysErrOld)
		{
			//printf("Save\n");
			//save
			this->s_mI2HWeight = this->mI2HWeight;
			this->s_mH2OWeight = this->mH2OWeight;

			sysErrOld = sysErrNew;
			if(changeEta)
			{
				float newEta = this->eta * 1.05;
				this->eta = newEta < 0.9 ? newEta : this->eta ;
			}
			

		}
		else if(sysErrNew > sysErrOld * 1.04 )
		{
			//restore
			restore = true;
			//printf("Restore\n");
			this->mI2HWeight = this->s_mI2HWeight;
			this->mH2OWeight = this->s_mH2OWeight;
			//alpha = 0;
			
			CMatrix tempError(this->numOutput, this->numSample);
			Forward(this->mI2HWeight, this->mHideBias, this->mH2OWeight, this->mOutputBias, VERBOSE);
			mOutputError = mDemoOutput - this->mOutOutput;
			float sysErrSpe = mOutputError.GetSystemError();
			//printf("Special err = %.6f\n", sysErrSpe);

			if (changeEta)
			{
				float newEta = this->eta* 0.7;
				this->eta = newEta > 0.01 ? newEta : this->eta;
			}
			
		}

		if(maxtime != 0)
		{
			TIMEV_END(t);
			float nowtime = t* 1000;
			TIMEV_START(t);
			if(nowtime > maxtime)
			{
				break;
			}
		}
		
		
		#ifdef _DEBUG
printf("loop = %d, New = %.6f, Old = %.6f, eta = %.4f %s\n", nLoopTimes, sysErrNew, sysErrOld, this->eta, restore ? "---RESTORE" : "");
#endif // _DEBUG
		



	}
	TIMEV_END(t);

	printf("Iter = %d, Err = %.6f\n", nLoopTimes, sysErrNew);
	printf("Forward Count = %d\n", forwardCount);
	printf("Time = %f (ms)\n", t * 1000);
}

void MaNeural::TrainSet(Image* imageList, int count, float diff, int maxIter, bool changeEta, bool deltaWeight, float maxtime)
{

	float alpha = this->alpha;

	this->d_mI2HWeight.Resize(numHidden, numInput);
	this->d_mH2OWeight.Resize(numOutput, numHidden);

	this->d_mH2OWeight.m_pTMatrix.clear();
	this->d_mI2HWeight.m_pTMatrix.clear();

	

	for(int i=0; i < count; i++)
	{
		int length = imageList[i].length;
		for(int j=0;j<length;j++)
		{
			this->mInputValue.m_pTMatrix (j, i) = imageList[i].content[j] / 16;
			
		}
		

	}
	//cout << "First mInputValue" << endl;
	//this->mInputValue.Print();
	this->mDemoOutput.Resize(this->numOutput, this->numSample);
	for(int i=0;i < count;i++)
	{
		float* target = tarptr->GetTarget(imageList[i].label);
		for(int j = 0;j<this->numOutput;j++)
		{
			mDemoOutput.m_pTMatrix (j, i) = target[j];
		}
	}


	__TrainSet(count, diff, maxIter, changeEta, deltaWeight, maxtime);
}

void MaNeural::PrintTest(float* input)
{
	this->numSample = 1;
	this->mInputValue.Resize(this->numInput, 1);

	for(int i=0;i<this->numInput;i++)
	{
		this->mInputValue.m_pTMatrix(i, 0) = input[i];
		cout << this->mInputValue.m_pTMatrix(i, 0) << endl;
		
	}
	Forward(this->mI2HWeight, this->mHideBias, this->mH2OWeight, this->mOutputBias, false);

	//this->mDemoOutput.Print();
	printf("Real:\n");
	this->mOutOutput.Print();
	printf("\n");
}


void MaNeural::Forward(CMatrix& _mI2HWeight, CMatrix& _mHideBias, CMatrix& _mH2OWeight, CMatrix& _mOutputBias, bool verbose)
{
	forwardCount++;
	//printf("Forward ...\n");
	CMatrix cMExHideBias;
	cMExHideBias.nncpyi(_mHideBias, this->numSample);
	//cout << "_mHideBias" << endl;
	//_mHideBias.Print();
	//cout << "cMExHideBias" << endl;
	//cMExHideBias.Print();


	CMatrix cMHidePureInput(this->numHidden, this->numSample);
	
	if(verbose)
	{
		cout << "mI2HWeight" << endl;
		_mI2HWeight.Print();
		cout << "mInputValue" << endl;
		this->mInputValue.Print();
	}

	
	cMHidePureInput = _mI2HWeight * this->mInputValue;
	//cout << "cMHidePureInput" << endl;
	//cMHidePureInput.Print();


	cMHidePureInput += cMExHideBias;


	//cout << "cMHidePureInput" << endl;
	//cMHidePureInput.Print();
	//CMatrix cMHideOutput(this->numHidden, this->numSample);

	this->mHideOutput = cMHidePureInput.Sigmoid();
	if (verbose)
	{
		cout << "_mH20Weight" << endl;
		_mH2OWeight.Print();
		cout << "mHideOutput" << endl;
		this->mHideOutput.Print();
	}
	

	CMatrix cMExOutputBias;
	//cout << "OutputBias" << endl;
	//_mOutputBias.Print();
	cMExOutputBias.nncpyi(_mOutputBias, this->numSample);
	//cout << "cMExOutputBias" << endl;
	//cMExOutputBias.Print();

	CMatrix cMOutPureInput(this->numOutput, this->numSample);
	
	//cout << "mHideOutput" << endl;
	//this->mHideOutput.Print();
	if(verbose)
	{
		
	}
	

	cMOutPureInput = _mH2OWeight * this->mHideOutput;
	//cout << "cMOutPureInput" << endl;
	//cMOutPureInput.Print();

	cMOutPureInput += cMExOutputBias;


	//cout << "cMOutPureInput" << endl;
	//cMOutPureInput.Print();
	this->mOutOutput = cMOutPureInput.Sigmoid();

	if (verbose)
	{
		cout << "mOutOutput" << endl;
		this->mOutOutput.Print();
	}
	
}


bool MaNeural::Test(float* input, int label)
{
	this->numSample = 1;
	this->mInputValue.Resize(this->numInput, 1);

	for(int i=0;i<this->numInput;i++)
	{
		this->mInputValue.m_pTMatrix(i, 0) = input[i];
	}
	Forward(this->mI2HWeight, this->mHideBias, this->mH2OWeight, this->mOutputBias);

	float* output = new float[this->numOutput];
	for(int i=0;i<this->numOutput;i++)
	{
		output[i] = this->mOutOutput.m_pTMatrix(i, 0);
	}

	int predict = tarptr->Check(output);

	printf("%d -> %d\n", label, predict);
	delete[] output;
	return label == predict;
}

void MaNeural::TestSet(Image* imageList, int count)
{
	int right = 0;
	for(int i=0;i<count;i++)
	{
		if(Test(imageList[i].content, imageList[i].label))
		{
			right++;
		}
	}
	printf("Correct percent = %.3f%%\n", ((float)(right*100))/count);
}