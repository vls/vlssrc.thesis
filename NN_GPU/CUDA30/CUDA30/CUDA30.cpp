// CUDA30.cpp : 定义控制台应用程序的入口点。
//
#include <stdio.h>
#include "Sample.h"


int main(int argc, char** argv)
{
	printf("Hello World\n");

	if (!InitCUDA())
	{
		printf("Can't init CUDA\n");
		return 0;
	}

	getchar();
	return 0;
}

