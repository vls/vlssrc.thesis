#include "Neural.h"
#include <iostream>
#include <math.h>
#include <time.h>
#include <string.h>

using namespace std;



float Sigmoid(float x)
{
    return 1 / (1 + exp(0 - x));
}

Neural::Neural() : layers(NULL), InputLayer(NULL), OutputLayer(NULL)
{
    this->units = UNITS;
    this->num_layer = NUM_LAYERS;
    this->eta = LEARNCOST;

    this->tarptr = new TBinGen(this->units[this->num_layer -1], OUTPUT);
}

Neural::Neural(int* units, int num_layer, float learnRate, TargetGenBase* tarptr) : layers(NULL), InputLayer(NULL), OutputLayer(NULL)
{
    this->units = units;
    this->num_layer = num_layer;

    this->eta = learnRate;

    this->tarptr = tarptr;
}

Neural::~Neural()
{
    if(this->layers != NULL)
    {
        for(int i = 0; i < NUM_LAYERS; i++)
        {
            if(this->layers[i] != NULL)
                delete this->layers[i];
        }
        delete this->layers;
    }


}

void Neural::InitLayer()
{
    cout << "Init Layer" << endl;
    for(int i = 0;i < NUM_LAYERS; i++)
    {
        int lowerUnits = i == 0 ? 0 : this->layers[i-1]->units;

        this->layers[i] = new Layer(UNITS[i], lowerUnits);
    }
}

void Neural::Init()
{
    this->numInput = this->units[0];
    this->numOutput = this->units[this->num_layer-1];

    this->layers = new Layer*[NUM_LAYERS];

    InitLayer();

    this->InputLayer = this->layers[0];
    this->OutputLayer = this->layers[this->num_layer - 1];
	
	this->alpha = 1.05;
	this->beta = 0.7;

	this->lastErr = 0;
}

float Neural::Train(float* input, float* target)
{
	Simulate(input, target, true);
    return 0.0f;
}

void Neural::GenerateWeight()
{
    for(int l = 1; l < this->num_layer; l++)
    {
        for (int i = 0; i < this->layers[l]->units; i++)
        {
            for (int j = 0; j < this->layers[l-1]->units; j++)
            {
                this->layers[l]->weight[i][j] = GetRandom(-0.5, 0.5);

            }
			this->layers[l]->bias[i] = GetRandom(-0.5, 0.5);
        }
    }
}

float Neural::GetRandom(float low, float high)
{
    return ((float) rand() / RAND_MAX) * (high - low) + low;
}

void Neural::ForwardNet()
{
    for (int l = 0; l < this->num_layer -1; l++)
    {
        ForwardLayer(this->layers[l], this->layers[l+1]);
    }
}

void Neural::ForwardLayer(Layer* lower, Layer* upper)
{
    for(int i = 0; i < upper->units; i++)
    {
        float sum = 0;
        for(int j = 0; j < lower->units; j++)
        {
            sum += upper->weight[i][j] * lower->output[j];
            int zz = 0;
        }
		//sum += upper->bias[i];
        upper->output[i] = ::Sigmoid(sum);

        int zz = 0;
    }
}

void Neural::SetInput(float* input)
{
    for(int i=0;i< this->InputLayer->units; i++)
    {
        this->InputLayer->output[i] = input[i] == 0 ? LOW : HIGH;
    }
}

void Neural::Simulate(float* input, float* target, bool Training)
{
    SetInput(input);

	ForwardNet();
	ComputeNetError(target);
/*
	if(Training)
	{
		BackNet();
		AdjustWeight();
	}
*/
}

void Neural::ComputeNetError(float* target)
{
	//this->Error = 0;
	for(int i = 0; i < this->OutputLayer->units; i++)
	{

		float Out = this->OutputLayer->output[i];
		float Err = target[i] - Out;
		this->OutputLayer->error[i] = Out * (1 - Out) * Err;
		float delta = Err * Err;
		this->Error += delta;
		//printf("%.3f %.3f %.3f\n", Err, delta, this->Error);
	}
}

void Neural::BackNet()
{
	for (int l = this->num_layer - 1; l > 1; l--)
	{
		BackLayer(this->layers[l], this->layers[l-1]);

	}
}

void Neural::BackLayer(Layer* upper, Layer* lower)
{
	for(int i=0; i < lower->units; i++)
	{
		float Out = lower->output[i];
		float err = 0;
		for(int j = 0; j < upper->units; j++)
		{
			err += upper->weight[j][i] * upper->error[j];

		}
		lower->error[i] = Out * (1 - Out) * err;
	}
}

void Neural::AdjustWeight()
{
	for(int l = 1; l < this->num_layer; l++)
	{
		for(int i = 0; i<this->layers[l]->units; i++)
		{
		    float Err = this->layers[l]->error[i];
			for(int j = 0; j < this->layers[l-1]->units; j++)
			{
				float Out = this->layers[l-1]->output[j];

                float delta = this->eta * Err * Out;
				this->layers[l]->weight[i][j] += delta;
				int zz= 0;
			}
			float delta = this->eta * Err;
			this->layers[l]->bias[i] += delta;

			int zz= 0;
		}
	}
}

void Neural::TrainSet(Image* imageList, int count, float diff, int maxIter)
{
    printf("Start training... total samples = %d\n", count);
    double t = 0;
    TIMEV_START(t);
    for(int k=0;k< maxIter;k++)
    {
        //FileGuard guard("output.txt", "w");
        char buf[128];
        _strtime(buf);

        this->Error = 0;
        for(int i=0;i< count;i++)
        {



            Train(imageList[i].content, tarptr->GetTarget(imageList[i].label));





        }
        this->Error /= count;
		

        if(this->Error < diff)
        {
            TIMEV_END(t);
            printf("Train succeeded\n");
            printf("%s Last Error = %.6f, iter = %d\n", buf, this->Error, k);
            printf("Elapsed time = %.6f s\n", t);
            return;
        }

        printf("%s Error = %.6f, iter = %d\n", buf, this->Error, k);
    }

    printf("Train failed\n");
}

bool Neural::Test(float* input, int label)
{
    Simulate(input, this->tarptr->GetTarget(label), false);

    int predict = tarptr->Check(this->OutputLayer->output);

    printf("%d -> %d\n", label, predict);
    return label == predict;
}

void Neural::TestSet(Image* imageList, int count)
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





