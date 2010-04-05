#include "MaNeural.h"

#include <iostream>
const int _NUMLAYER = 3;
#define VERBOSE false
using namespace std;


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

}

void MaNeural::GenerateWeight()
{
	this->mI2HWeight.RandomInitialize(HIGH, LOW);
	this->mH2OWeight.RandomInitialize(HIGH, LOW);
	this->mHideBias.RandomInitialize(HIGH, LOW);
	this->mOutputBias.RandomInitialize(HIGH, LOW);
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

	TrainSet(count, diff, maxIter, changeEta, deltaWeight);
}

void MaNeural::TrainSet(int count, float diff, int maxIter, bool changeEta, bool deltaWeight)
{
	Forward(this->mI2HWeight, this->mHideBias, this->mH2OWeight, this->mOutputBias, VERBOSE);




	CMatrix mOutputError(this->numOutput, this->numSample);
	mOutputError = mDemoOutput - this->mOutOutput;

	cout << "mDemoOutput "<< endl;
	mDemoOutput.Print();
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

		// ��������deltaֵ
		// ע��: �˴�'/' �� '���'!!!
		CMatrix	mOutDelta (this->numOutput, this->numSample);
		mOutDelta = (this->mOutOutput - this->mOutOutput  / this->mOutOutput ) / mOutputError;
		//cMatrixOutputLayerDelta ��һ������������Ϊ���Ԫ��������Ϊ������

		CMatrix t_mH2OWeight (this->mH2OWeight.GetColCount() , this->mH2OWeight.GetRowCount());
		t_mH2OWeight = this->mH2OWeight.Transpose();

		// ���������deltaֵ
		// ע��: �˴�'/' �� '���'!!!
		CMatrix mHideDelta;
		mHideDelta.CopyMatrix ( (this->mHideOutput - (this->mHideOutput / this->mHideOutput)) / ( t_mH2OWeight * mOutDelta) );



		// �����µ�����㵽�������Ȩֵ
		CMatrix mNewI2HWeight (this->mI2HWeight.GetRowCount(), this->mI2HWeight.GetColCount());
		// ������µ�������ķ�ֵ
		CMatrix mNewHideBias (this->numHidden, this->numSample);
		// �����µ������㵽������Ȩֵ
		CMatrix mNewH2OWeight (this->mH2OWeight.GetRowCount(), this->mH2OWeight.GetColCount());
		// �����µ������ķ�ֵ
		CMatrix mNewOutputBias (this->numOutput, this->numSample);
		// �����µ�������
		CMatrix cMatrixNewOutputLayerError(this->numOutput, this->numSample);


		// Ȩֵ�ͷ�ֵ����
		mNewH2OWeight = mOutDelta * (this->mHideOutput.Transpose ()) * (this->eta);


		mNewOutputBias = mOutDelta * this->eta;//���﷧ֵ�ĸı�û��*nStep

		mNewI2HWeight = mHideDelta * (this->mInputValue.Transpose ()) * (this->eta);



		mNewHideBias = mHideDelta * this->eta;

		// ��ֵ
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



		CMatrix tempHideBias(this->numHidden, 1);
		mNewHideBias.CopySubMatrix(tempHideBias, 0, mNewHideBias.GetColCount() - 1);
		tempHideBias += this->mHideBias;

		CMatrix tempOutputBias(this->numOutput, 1);
		mNewOutputBias.CopySubMatrix(tempOutputBias, 0, mNewOutputBias.GetColCount() - 1);
		tempOutputBias += this->mOutputBias;





		//save delta weight
		if(deltaWeight)
		{
			this->d_mH2OWeight = mNewH2OWeight;
			this->d_mI2HWeight = mNewI2HWeight;
		}


		// ǰ�����
		Forward(this->mI2HWeight, tempHideBias, this->mH2OWeight, tempOutputBias, VERBOSE);


		cMatrixNewOutputLayerError = mDemoOutput - this->mOutOutput;;
		float sysErrNew =	cMatrixNewOutputLayerError.GetSystemError ();

		mOutputError = cMatrixNewOutputLayerError;

		alpha = this->alpha;



		if(sysErrNew < sysErrOld)
		{
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
			this->mI2HWeight = this->s_mI2HWeight;
			this->mH2OWeight = this->s_mH2OWeight;
			//alpha = 0;

			if (changeEta)
			{
				float newEta = this->eta* 0.7;
				this->eta *= newEta > 0.01 ? newEta : this->eta;
			}
			
		}

		printf("loop = %d, New = %.6f, Old = %.6f, eta = %.4f\n", nLoopTimes, sysErrNew, sysErrOld, this->eta);



	}
}

void MaNeural::TrainSet(Image* imageList, int count, float diff, int maxIter, bool changeEta, bool deltaWeight)
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


	TrainSet(count, diff, maxIter, changeEta, deltaWeight);
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
	printf("Forward ...\n");
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


	//cMHidePureInput += cMExHideBias;


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

	//cMOutPureInput += cMExOutputBias;


	//cout << "cMOutPureInput" << endl;
	//cMOutPureInput.Print();
	this->mOutOutput = cMOutPureInput.Sigmoid();

	if (verbose)
	{
		cout << "mOutOutput" << endl;
		this->mOutOutput.Print();
	}
	
}