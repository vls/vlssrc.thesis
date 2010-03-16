


#ifndef LAYER_H_INCLUDED
#define LAYER_H_INCLUDED

#include "global.h"

class Layer
{
public:
    int units;

    float* output;
    float* error;
    Array2Df weight;

	float* bias;

    Layer(int units, int lowerUnits);
    ~Layer();
};

class DLayer : public Layer
{
public:
    DLayer(int units, int lowerUnits);
    Array2Df dWeight;
};


#endif // LAYER_H_INCLUDED
