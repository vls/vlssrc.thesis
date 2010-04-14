#include "Image.h"

int iDivUp(int, int);
bool InitSample(int, int, int, int, float*, float*, float*, float*);
void train(float, int, int, int, int, int, float*, float*, float*, float*, bool, float);
void logsig(float*, float*, float*, int, int);


int run(int argc, char** argv);
float Test(float* h_SamInEx, int InDim, int HiddenUnitNum, int OutDim ,int SamNum, float* h_W1Ex, float* h_W2Ex, Image* imageList);

bool InitImage(int SamNum, int InDim, int OutDim, int HiddenUnitNum, float* h_SamInEx, float* h_SamOut, float* h_W1Ex,float* h_W2Ex, Image* imageList);
int runImage(int argc, char** argv, Image* imageList, int trainnum, int testnum, int maxIter, bool, float);


void Print(float*, int row, int col, const char*);
void PrintHost(float* arr, int row, int col, const char* str);