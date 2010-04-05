#ifndef READER_H_INCLUDED
#define READER_H_INCLUDED

#include "global.h"
#include "Image.h"
#include <stdio.h>
bool read(const char* filename, Image* imageList, int maxCount);
bool read13(const char* filename, Image* imageList, int maxCount);
bool read64(const char* filename, Image* imageList, int maxCount);

class FileGuard
{
public:
    FILE* fp;

    FileGuard(const char* filename, const char* mode);

    ~FileGuard();

};

#endif
