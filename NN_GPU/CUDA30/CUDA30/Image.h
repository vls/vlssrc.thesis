#ifndef IMAGE_H_INCLUDED
#define IMAGE_H_INCLUDED

class Image
{

public:
	int label;
	int length;
	float* content;

	Image() : inited(false){}

	~Image();

	void Init(int length);

private:
	bool inited;
};

#endif
