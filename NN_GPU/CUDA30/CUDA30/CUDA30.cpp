// CUDA30.cpp : �������̨Ӧ�ó������ڵ㡣
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

