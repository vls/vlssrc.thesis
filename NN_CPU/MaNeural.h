#ifndef MANEURAL_H_INCLUDED
#define MANEURAL_H_INCLUDED

#include "MNeural.h"


class MaNeural : public MNeural
{
public:
	MaNeural(int* units, float learnRate, TargetGenBase* tarptr);
	virtual void TrainSet(Image* imageList, int count, float diff, int maxIter, float nStep, float maxStep);


	virtual bool Test(float* input, int label);
	virtual void TestSet(Image* imageList, int count);

	void Test();


protected:

	//void Forward(CMatrix& mI2HWeight, CMatrix& mHideBias, CMatrix& mH2OWeight, CMatrix& mOutputBias);

	float alpha;
private:
	MaNeural(){}

};

#endif