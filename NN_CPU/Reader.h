#ifndef READER_H_INCLUDED
#define READER_H_INCLUDED

#include "global.h"
#include "Image.h"
#include <stdio.h>
float* read(const char* filename, Image* imageList, int maxCount);

class FileGuard
{
public:
    FILE* fp;

    FileGuard(const char* filename, const char* mode);

    ~FileGuard();

};

#endif
