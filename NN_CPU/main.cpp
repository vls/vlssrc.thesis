#include <iostream>
#include "Neural.h"
#include "Layer.h"
#include "Reader.h"
#include "Image.h"
#include "TargetGen.h"
using namespace std;

const int READNUM = 50;
const int TRAINNUM = 30;

void Neural::Test()
{
	const int NUMLAYER = 3;
	int _UNITS[NUMLAYER] = {3, 2, 1};
	TBinGen gen(4, 10);

	Neural nn(_UNITS, NUMLAYER, 0.9, &gen);

	nn.layers[1]->weight[0][0] = 0.2;
	nn.layers[1]->weight[0][1] = -0.4;
	nn.layers[1]->weight[0][2] = -0.5;

	nn.layers[1]->weight[1][0] = -0.3;
	nn.layers[1]->weight[1][1] = 0.1;
	nn.layers[1]->weight[1][2] = 0.2;


	nn.layers[2]->weight[0][0] = -0.3;
	nn.layers[2]->weight[0][1] = -0.2;

	nn.layers[1]->bias[0] = -0.4;
	nn.layers[1]->bias[1] = 0.2;
	nn.layers[2]->bias[0] = 0.1;

	float input[3] = {1.0, 0.0 , 1.0};
	float target[1] = {1.0};
	nn.Train(input, target);
}

int main()
{
    cout << "Hello world!" << endl;
	Image imageList[READNUM];


    read("image.txt", imageList, READNUM);

    TBinGen gen(4, 10);

    Neural* nn = new DNeural();
    nn->Init();
    nn->tarptr = &gen;

    nn->GenerateWeight();
    nn->TrainSet(imageList, TRAINNUM, 0.0001, 1000);
    nn->TestSet(imageList + TRAINNUM, READNUM - TRAINNUM);
    delete nn;



    cout << "Done" << endl;
    return 0;
}
