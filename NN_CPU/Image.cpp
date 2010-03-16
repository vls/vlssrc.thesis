#include "Image.h"




Image::~Image()
{
	if (this->inited)
	{
		delete[] this->content;
	}
}

void Image::Init(int length)
{
	if(!inited)
	{
		this->inited = true;
		this->length = length;
		this->content = new float[this->length];
	}
}

