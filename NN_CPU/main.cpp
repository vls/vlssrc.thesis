#include <iostream>
#include "Neural.h"
#include "Layer.h"
#include "Reader.h"
#include "Image.h"
#include "TargetGen.h"
#include <string.h>
#include "Matrix.h"
#include "MNeural.h"
#include "MaNeural.h"
#include "Tools.h"
#include <cutil.h>
using namespace std;

int READNUM = 100;
const int TRAINNUM = 20;
const double PI = 3.1415926;

void Neural::Test()
{
	const int NUMLAYER = 3;
	int _UNITS[NUMLAYER] = {3, 2, 1};
	TBinGen gen(4, 10);

	Neural nn(_UNITS, NUMLAYER, 0.9, &gen);

	nn.layers[1]->weight[0][0] = 0.2f;
	nn.layers[1]->weight[0][1] = -0.4f;
	nn.layers[1]->weight[0][2] = -0.5f;

	nn.layers[1]->weight[1][0] = -0.3f;
	nn.layers[1]->weight[1][1] = 0.1f;
	nn.layers[1]->weight[1][2] = 0.2f;


	nn.layers[2]->weight[0][0] = -0.3f;
	nn.layers[2]->weight[0][1] = -0.2f;

	nn.layers[1]->bias[0] = -0.4f;
	nn.layers[1]->bias[1] = 0.2f;
	nn.layers[2]->bias[0] = 0.1f;

	float input[3] = {1.0, 0.0 , 1.0};
	float target[1] = {1.0};
	nn.Train(input, target);
}

void MatrixNeuralTest()
{
	int nSample = 3;
	MyArray<float> x(nSample);
	MyArray<float> y(nSample);
	float* xptr = x.GetPtr();
	float* yptr = y.GetPtr();
	
	for(int i=0; i<nSample; i++)
	{
		x[i] = (double)i / (double)nSample;
		//用正弦曲线验证
		y[i] = 0.5 * sin(x[i]*PI*0.5 ) + 0.5;
		//y[i][0] = -0.4 * x[i][0] + 1;
		//y[i][0] = 0.25*(x[i][0]-2)*(x[i][0]-2);
	}

	int units[3];
	units[0] = 1;
	units[1] = 8;
	units[2] = 1;

	
	MaNeural* nptr = new MaNeural(&units[0], 0.1, NULL);
	nptr->Init(nSample);
	nptr->GenerateWeight();
	nptr->TrainSet(xptr, yptr, TRAINNUM, 0.001, 5000, true, true);
	for(int i=0;i<nSample;i++)
	{
		nptr->PrintTest(xptr+i);
	}
}

void TestInverse()
{
	CMatrix c1(2,2);
	c1.m_pTMatrix(0, 0) = 1;
	c1.m_pTMatrix(0, 1) = 3;
	c1.m_pTMatrix(1, 0) = 2;
	c1.m_pTMatrix(1, 1) = 5;

	c1.Print();
	c1.Inverse().Print();
}

int main(int argc, char** argv)
{
    cout << "Hello world!" << endl;
    MaNeural* nn = NULL;


	//freopen("out.txt", "w", stdout);
	//freopen("out.txt", "w", stderr);
	//MatrixNeuralTest();


	int iter = 4000;
	int trainnum = 500;
	float precision = 0.000001;

#ifdef NDEBUG
	cutGetCmdLineArgumenti(argc,(const char**) argv, "iter", &iter);
	cutGetCmdLineArgumenti(argc,(const char**) argv, "train", &trainnum);
	cutGetCmdLineArgumentf(argc,(const char**) argv, "prec", &precision);
#endif // NDEBUG

	printf("Iter = %d\n", iter);
	printf("TrainNum = %d\n", trainnum);

    try
    {
		int readnum = trainnum + trainnum /2;
        Image* imageList = new Image[readnum];
	
		
		if(read64("my_optdigits.tra", imageList, readnum))
		{
			printf("Read samples succeeded. Initializing network...\n");
			int units[3];
			units[0] = 64;
			units[1] = 16;
			units[2] = 4;

			TBinGen gen(4, 10);
			MaNeural* nptr = new MaNeural(&units[0], 0.1, &gen);
			nptr->Init(trainnum);
			nptr->GenerateWeight();
			double t = 0;
			printf("Initialized. Begin to work...\n");
			TIMEV_START(t);
			nptr->TrainSet(imageList, trainnum, 0.000001, iter, true, false);
			TIMEV_END(t);
			printf("time = %.6f\n", t * 1000);

			int testNum = min(readnum - trainnum, trainnum / 2);
			//nptr->TestSet(imageList+TRAINNUM, testNum);
		}
		
		delete[] imageList;
		
    }
    catch(string errs)
    {
        cerr << errs << endl;
    }

    if(nn!= NULL)
        delete nn;
	freopen("CON", "w", stdout);
    cout << "Press any key to EXIT..." << endl;
	getchar();
    return 0;
}




/*
if(read("image_21x21.txt", imageList, READNUM))
{

TBinGen gen(4, 10);
int units[3];
units[0] = 441;
units[1] = 21;
units[2] = 4;
nn = new MaNeural(&units[0], LEARNCOST, &gen);
nn->Init(TRAINNUM);
nn->tarptr = &gen;

nn->GenerateWeight();

nn->TrainSet(imageList, TRAINNUM, 0.0001, 10000);
//nn->TestSet(imageList + TRAINNUM, READNUM - TRAINNUM);




}

*/

/*
if(read("image_21x21.txt", imageList, READNUM))
{
TBinGen gen(4, 10);
Neural* nptr = new DNeural();
nptr->Init();
nptr->GenerateWeight();
nptr->TrainSet(imageList, TRAINNUM, 0.0001, 10000);
nptr->TestSet(imageList+TRAINNUM, READNUM - TRAINNUM);
printf("Test Num = %d\n", READNUM - TRAINNUM);
}
*/