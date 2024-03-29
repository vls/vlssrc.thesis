#ifndef MNEURAL_H_INCLUDED
#define MNEURAL_H_INCLUDED

#include "Neural.h"
#include "Matrix.h"

class MNeural
{
public:
    MNeural();

    MNeural(int* units, float learnRate, TargetGenBase* tarptr);

    virtual ~MNeural();

    virtual void Init(int);


    virtual void TrainSet(Image* imageList, int count, float diff, int maxIter, float nStep, float maxStep);
    virtual void GenerateWeight();


    virtual bool Test(float* input, int label);
    virtual void TestSet(Image* imageList, int count);

    void Test();

	TargetGenBase* tarptr;
protected:

    void Forward(CMatrix& mI2HWeight, CMatrix& mHideBias, CMatrix& mH2OWeight, CMatrix& mOutputBias);

	int* units;
    int numSample;
    int numHidden;
	int numInput;
	int numOutput;

	float eta;
    CMatrix mInputValue;
    CMatrix mI2HWeight;
    CMatrix mHideBias;
    CMatrix mH2OWeight;
    CMatrix mOutputBias;

    CMatrix mHideOutput;
    CMatrix mOutOutput;
};

#endif // MNEURAL_H_INCLUDED
