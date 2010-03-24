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

			imageList[count].length = N;

			for(int i=0;i < N; i++)
			{
				fscanf(guard.fp, "%d", &imageList[count].content[i]);
			}

			int zz = 0;
		}
        return true;
	}
	else
		cout << "Not Found"  << endl;
	return false;

}
