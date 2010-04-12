#include "Image.h"

int iDivUp(int, int);
bool InitSample(int, int, int, int, float*, float*, float*, float*);
void train(float, int, int, int, int, int, float*, float*, float*, float*, bool);
void logsig(float*, float*, float*, int, int);


int run(int argc, char** argv);


bool InitImage(int SamNum, int InDim, int OutDim, int HiddenUnitNum, float* h_SamInEx, float* h_SamOut, float* h_W1Ex,float* h_W2Ex, Image* imageList);
int runImage(int argc, char** argv, Image* imageList, int count, int maxIter, bool);