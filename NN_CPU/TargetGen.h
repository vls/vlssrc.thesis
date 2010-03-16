#ifndef TARGETGEN_H_INCLUDED
#define TARGETGEN_H_INCLUDED

#include "global.h"

class TargetGenBase
{
public:
    virtual int Check(float* output) = 0;
    virtual float* GetTarget(int label);
protected:
    float** m_target;
    virtual ~TargetGenBase(){};
};

class T10Gen : public TargetGenBase
{
public:
    T10Gen(int numOutput);
    virtual int Check(float* output);
    virtual ~T10Gen();
private:
    int numOutput;
};

class TBinGen : public TargetGenBase
{
    public:
    TBinGen(int numOutput, int numClass);
    virtual int Check(float* output);
    virtual ~TBinGen();
private:
    int numOutput;
    int numClass;
};

#endif // TARGETGEN_H_INCLUDED
