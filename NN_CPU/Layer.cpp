#include "Layer.h"

#include <iostream>

using namespace std;

Layer::Layer(int units, int lowerUnits) : weight(boost::extents[units][lowerUnits])
{
    this->units = units;
    this->output = new float[units];
    this->error = new float[units];
	this->bias = new float[units];
}

Layer::~Layer()
{
	delete[] bias;
    delete[] output;
    delete[] error;
}


DLayer::DLayer(int units, int lowerUnits) : Layer(units, lowerUnits), dWeight(boost::extents[units][lowerUnits])
{
}
