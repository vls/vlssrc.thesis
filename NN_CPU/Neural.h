#ifndef NEURAL_H_INCLUDED
#define NEURAL_H_INCLUDED


#include "global.h"
#include "Layer.h"
#include <math.h>
#include "Image.h"
#include "reader.h"
#include "TargetGen.h"

const float HIGH = 1.0f;
const float LOW = 0.0f;

class Neural
{


public:
    Neural();

    Neural(int* units, int num_layer, float learnRate, TargetGenBase* tarptr);

    virtual ~Neural();

    virtual void InitLayer();
    virtual void Init();


    virtual void TrainSet(Image* imageList, int count, float diff, int maxIter);
    virtual void GenerateWeight();


    virtual bool Test(float* input, int label);
    virtual void TestSet(Image* imageList, int count);

    float Error;

    void Test();

    TargetGenBase* tarptr;



protected:
    virtual float Train(float* input, float* target);

    virtual void SetInput(float* input);

    Layer** layers;

    int numInput;
    int numOutput;

    float eta;
	float alpha;
	float beta;

	float lastErr;

    int* units;
    int num_layer;

    Layer* InputLayer;
    Layer* OutputLayer;

    float GetRandom(float low, float high);

	void ForwardNet();

	void ForwardLayer(Layer* lower, Layer* upper);

	void Simulate(float* input, float* target, bool Training);

	void ComputeNetError(float* target);

	void BackNet();
	void BackLayer(Layer* upper, Layer* lower);

	virtual void AdjustWeight();
};

float Sigmoid(float x);


class DNeural : public Neural
{
public:
    
	DNeural(int* units, int num_layer, float learnRate, TargetGenBase* tarptr);
    virtual ~DNeural(){};

private:
	virtual void InitLayer();
    virtual void AdjustWeight();

};

#endif // NEURAL_H_INCLUDED
