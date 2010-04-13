#ifndef MANEURAL_H_INCLUDED
#define MANEURAL_H_INCLUDED

#include "MNeural.h"


class MaNeural
{
public:
	MaNeural(int* units, float learnRate, TargetGenBase* tarptr);

	virtual ~MaNeural();

	virtual void Init(int);
	virtual void GenerateWeight();

	virtual void TrainSet(Image* imageList, int count, float diff, int maxIter)
	{
		TrainSet(imageList, count, diff, maxIter, false, false);
	}
	void TrainSet(float* input, float* target, int count, float diff, int maxIter, bool changeEta, bool deltaWeight);
	
	void TrainSet(Image* imageList, int count, float diff, int maxIter, bool changeEta, bool deltaWeight)
	{

		TrainSet(imageList, count, diff, maxIter,  changeEta, deltaWeight, 0.0f);
	}

	void TrainSet(Image* imageList, int count, float diff, int maxIter, bool changeEta, bool deltaWeight, float maxtime);


	void Forward(CMatrix& _mI2HWeight, CMatrix& _mHideBias, CMatrix& _mH2OWeight, CMatrix& _mOutputBias)
	{
		Forward(_mI2HWeight, _mHideBias, _mH2OWeight, _mOutputBias, false);
	}
	void Forward(CMatrix& _mI2HWeight, CMatrix& _mHideBias, CMatrix& _mH2OWeight, CMatrix& _mOutputBias, bool verbose);
	void Test();

	void PrintTest(float* input);

	virtual bool Test(float* input, int label);
	virtual void TestSet(Image* imageList, int count);

	TargetGenBase* tarptr;
protected:
	void __TrainSet(int count, float diff, int maxIter, bool changeEta, bool deltaWeight)
	{
		__TrainSet(count, diff, maxIter, changeEta, deltaWeight, 0.0f);
	}
	void __TrainSet(int count, float diff, int maxIter, bool changeEta, bool deltaWeight, float maxtime);
	//void Forward(CMatrix& mI2HWeight, CMatrix& mHideBias, CMatrix& mH2OWeight, CMatrix& mOutputBias);

	int* units;
	int numSample;
	int numHidden;
	int numInput;
	int numOutput;

	float alpha;
	float beta;
	float eta;

	CMatrix mInputValue;
	CMatrix mI2HWeight;
	CMatrix mHideBias;
	CMatrix mH2OWeight;
	CMatrix mOutputBias;

	CMatrix mHideOutput;
	CMatrix mOutOutput;
private:
	MaNeural(){}
	CMatrix mDemoOutput;
	CMatrix d_mI2HWeight;
	CMatrix d_mH2OWeight;
	
	CMatrix s_mI2HWeight;
	CMatrix s_mH2OWeight;

};

#endif