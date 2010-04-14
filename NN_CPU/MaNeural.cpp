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

	this->mInputValue.Resize(numSample, numInput+1);

	this->mI2HWeight.Resize(numInput+1, numHidden);

	this->mHideOutput.Resize(numSample, numHidden);
	this->mHideOutEx.Resize(numSample, numHidden + 1);
	this->mH2OWeight.Resize(numHidden+1, numOutput);
	this->mOutOutput.Resize(this->numSample, this->numOutput);

}

void MaNeural::GenerateWeight()
{
	FILE* fptr = fopen("w1.txt", "r");
	FILE* fptr2 = fopen("w2.txt", "r");
	if(fptr != NULL && fptr2 != NULL)
	{
		printf("Weight files exist, reading files...\n");
		for(int i=0; i<numInput+1; i++)
		{
			for(int j=0;j<numHidden;j++)
			{
				float val;
				fscanf(fptr, "%f", &val);
				this->mI2HWeight.m_pTMatrix(i,j)= val;
			}
		}

		for(int i=0;i<numHidden+1;i++)
		{
			for(int j=0;j<numOutput;j++)
			{
				float val;
				fscanf(fptr2, "%f", &val);
				this->mH2OWeight.m_pTMatrix(i,j) = val;
			}
		}
		fclose(fptr);
		fclose(fptr2);
	}
	else
	{
		printf("Files not found completely, generating...\n");
		this->mI2HWeight.RandomInitialize(HIGH, LOW);
		this->mH2OWeight.RandomInitialize(HIGH, LOW);
	}
	
	
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
	CMatrix W2(this->numHidden, this->numOutput);

	CMatrix mOutputError(this->numOutput, this->numSample);
	double t = 0;
	TIMEV_START(t);

	for(int i=0;i<numHidden;i++)
	{
		for(int j=0;j<numOutput;j++)
		{
			W2.m_pTMatrix(i,j) = this->mH2OWeight.m_pTMatrix(i,j);
		}
	}
	
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

		CMatrix delta2 = (this->mOutOutput - this->mOutOutput / this->mOutOutput) / mOutputError; //Sam*Out

		//delta2.Print("err");
		Sgemm(eta, this->mHideOutEx.Transpose(), delta2, 1.0f, this->mH2OWeight);
		
		
		
		CMatrix delta1(this->numSample, this->numHidden);
		Sgemm(1.0f, delta2, W2.Transpose(), 0.0f, delta1);



		CMatrix delta1_after = (this->mHideOutput - this->mHideOutput / this->mHideOutput) / delta1;
		//this->mI2HWeight.Print("W1 before");
		Sgemm(eta, this->mInputValue.Transpose(), delta1_after, 1.0f, this->mI2HWeight);
		//this->mI2HWeight.Print("W1 after");

		for(int i=0;i<numHidden;i++)
		{
			for(int j=0;j<numOutput;j++)
			{
				W2.m_pTMatrix(i,j) = this->mH2OWeight.m_pTMatrix(i,j);
			}
		}

		

		// Ç°Ïò¼ÆËã
		Forward(this->mI2HWeight, this->mHideBias, this->mH2OWeight, this->mOutputBias, VERBOSE);


		mOutputError = mDemoOutput - this->mOutOutput;

		sysErrNew =	mOutputError.GetSystemError ();

		alpha = this->alpha;


		bool restore = false;
		if(sysErrNew < sysErrOld)
		{
			//printf("Save\n");
			//save
			//this->s_mI2HWeight = this->mI2HWeight;
			//this->s_mH2OWeight = this->mH2OWeight;

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
			//this->mI2HWeight = this->s_mI2HWeight;
			//this->mH2OWeight = this->s_mH2OWeight;
			//alpha = 0;
			/*
			CMatrix tempError(this->numOutput, this->numSample);
			Forward(this->mI2HWeight, this->mHideBias, this->mH2OWeight, this->mOutputBias, VERBOSE);
			mOutputError = mDemoOutput - this->mOutOutput;
			float sysErrSpe = mOutputError.GetSystemError();
			*/
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
			this->mInputValue.m_pTMatrix (i,j) = imageList[i].content[j] / 16;
			
		}
		this->mInputValue.m_pTMatrix(i, length) = 1.0f;
		

	}
	//cout << "First mInputValue" << endl;
	//this->mInputValue.Print();
	this->mDemoOutput.Resize(this->numSample, this->numOutput);
	for(int i=0;i < count;i++)
	{
		float* target = tarptr->GetTarget(imageList[i].label);
		for(int j = 0;j<this->numOutput;j++)
		{
			mDemoOutput.m_pTMatrix (i, j) = target[j];
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


	CMatrix cMHidePureInput(this->numHidden, this->numSample);

	
	Sgemm(1.0f, this->mInputValue, this->mI2HWeight, 0.0f, this->mHideOutput);
	/*
	this->mInputValue.Print();
	this->mI2HWeight.Print();
	
	this->mHideOutput.Print();
	*/
	this->mHideOutput.CopyTo(this->mHideOutEx, 0, 0);
	this->mHideOutput.Sigmoid1();
	this->mHideOutEx.SigmoidEx1();
	
	Sgemm(1.0f, this->mHideOutEx, this->mH2OWeight, 0.0f, this->mOutOutput);
	
	this->mOutOutput.Sigmoid1();
	
}


bool MaNeural::Test(float* input, int label)
{
	this->numSample = 1;
	this->mInputValue.Resize( numSample, this->numInput+1);

	for(int i=0;i<this->numInput;i++)
	{
		this->mInputValue.m_pTMatrix(0, i) = input[i];
	}

	this->mInputValue.m_pTMatrix(0, this->numInput) = 1.0f;
	Forward(this->mI2HWeight, this->mHideBias, this->mH2OWeight, this->mOutputBias);

	float* output = new float[this->numOutput];
	for(int i=0;i<this->numOutput;i++)
	{
		output[i] = this->mOutOutput.m_pTMatrix(0, i);
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