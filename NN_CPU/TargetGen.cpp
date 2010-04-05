#include "TargetGen.h"
#include "Neural.h"


float* TargetGenBase::GetTarget(int label)
{
    if(this->m_target != NULL)
        return this->m_target[label];
    else
        return NULL;
}

T10Gen::T10Gen(int numOutput)
{
    this->numOutput = numOutput;

    this->m_target = new float*[numOutput];
    for(int i=0;i<this->numOutput ; i++)
    {
        this->m_target[i] = new float[numOutput];
        for(int j=0;j<this->numOutput;j++)
            this->m_target[i][j] = i ==j ? HIGH : LOW;
    }
}

int T10Gen::Check(float* output)
{
    float max = output[0];
    int maxpos = 0;
    for(int i=1;i<numOutput;i++)
    {
        if(output[i] > max)
        {
            max = output[i];
            maxpos = i;
        }
    }

    return maxpos;
}

T10Gen::~T10Gen()
{
    for(int i=0;i<this->numOutput ; i++)
    {
        delete[] this->m_target[i];
    }
    delete[] this->m_target;
}

TBinGen::TBinGen(int numOutput, int numClass)
{
    this->numOutput = numOutput;
    this->numClass = numClass;

    this->m_target = new float*[this->numClass];
    for(int i=0;i<this->numClass ; i++)
    {
        this->m_target[i] = new float[numOutput];
        for(int j=0;j<this->numOutput;j++)
            this->m_target[i][j] = LOW;

        int label = i;
        int count = this->numOutput -1;
        while(label)
        {
            if(label & 1)
                this->m_target[i][count] = HIGH;
            label >>= 1;
            count--;
        }
    }

    int zz = 0;
}

int TBinGen::Check(float* output)
{
    int predict = 0;
    float door = (HIGH - LOW) * 0.7 + LOW;
    for(int i=0;i<numOutput;i++)
    {
        predict <<= 1;
        if(output[i] > door)
        {
            predict |= 1;

        }
		printf("%.3f\t", output[i]);

    }
	printf("\n");



    return predict;
}

TBinGen::~TBinGen()
{
    for(int i=0;i<this->numClass ; i++)
    {
        delete[] this->m_target[i];
    }
    delete[] this->m_target;
}

