#include "Reader.h"

#include <iostream>
#include <stdio.h>

using namespace std;

FileGuard::FileGuard(const char* filename, const char* mode)
{
    this->fp = fopen(filename, mode);
}

FileGuard::~FileGuard()
{
    if(this->fp != NULL)
    {
        if(fclose(this->fp) != 0)
        {
            printf("Close file filed");
        }
    }
}

bool read(const char* filename, Image* imageList, int maxCount)
{
	FileGuard guard(filename, "r");

	if(guard.fp != NULL)
	{
		int total;
		int row, col;
		fscanf(guard.fp, "%d", &total);
		fscanf(guard.fp, "%d%d", &row, &col);
		cout << total << endl << row << endl << col << endl;

		int MAX = min(maxCount, total);

		for(int count = 0; count < MAX; count ++)
		{

			const int N = row * col;

			imageList[count].Init(N);
			fscanf(guard.fp, "%d", &imageList[count].label);
			//cout << imageList[count].label << endl;
			imageList[count].length = N;

			for(int i=0;i < N; i++)
			{
				fscanf(guard.fp, "%f", &imageList[count].content[i]);
				//cout << imageList[count].content[i] << "\t";
			}
			//cout << endl;
			int zz = 0;
		}
        return true;
	}
	else
		cout << "Not Found"  << endl;
	return false;

}


bool read13(const char* filename, Image* imageList, int maxCount)
{
	FileGuard guard(filename, "r");

	if(guard.fp != NULL)
	{
		int total;
		int vectorLen;
		fscanf(guard.fp, "%d", &total);
		fscanf(guard.fp, "%d", &vectorLen);
		cout << total << endl << vectorLen << endl;

		int MAX = min(maxCount, total);

		for(int count = 0; count < MAX; count ++)
		{

			const int N = vectorLen;

			imageList[count].Init(N);
			fscanf(guard.fp, "%d", &imageList[count].label);

			imageList[count].length = N;

			for(int i=0;i < N; i++)
			{
				fscanf(guard.fp, "%f", &imageList[count].content[i]);
			}

			int zz = 0;
		}
		return true;
	}
	else
		cout << "Not Found"  << endl;
	return false;
}

bool read64(const char* filename, Image* imageList, int maxCount)
{
	FileGuard guard(filename, "r");

	if(guard.fp != NULL)
	{

		const int MAX = 64;

		for(int count = 0; count < maxCount; count ++)
		{

			imageList[count].Init(MAX);
			

			imageList[count].length = MAX;

			for(int i=0;i < MAX; i++)
			{
				fscanf(guard.fp, "%f", &imageList[count].content[i]);
			}
			fscanf(guard.fp, "%d", &imageList[count].label);
		}
		
		return true;
	}
	else
		cout << "Not Found"  << endl;
	return false;
}