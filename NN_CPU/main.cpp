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

using namespace std;

const int READNUM = 50;
const int TRAINNUM = 5;

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
    MNeural* nn = NULL;


	//freopen("out.txt", "w", stdout);
	//freopen("out.txt", "w", stderr);

    try
    {
		CMatrix c1(2,2);
		c1.m_pTMatrix(0, 0) = 1;
		c1.m_pTMatrix(0, 1) = 3;
		c1.m_pTMatrix(1, 0) = 2;
		c1.m_pTMatrix(1, 1) = 5;

		c1.Print();
		c1.Inverse().Print();


        Image imageList[READNUM];
		
		
		
        if(read("image_21x21.txt", imageList, READNUM))
        {
			
            TBinGen gen(4, 10);
			int units[3];
			units[0] = 441;
			units[1] = 20;
			units[2] = 4;
            nn = new MaNeural(&units[0], 0.9, &gen);
            nn->Init(TRAINNUM);
            nn->tarptr = &gen;

            nn->GenerateWeight();
			
            nn->TrainSet(imageList, TRAINNUM, 0.0001, 10000, 0.001 ,10e6);
            //nn->TestSet(imageList + TRAINNUM, READNUM - TRAINNUM);
			

			

		}
		/*
		

		if(read("image_21x21.txt", imageList, READNUM))
		{
			TBinGen gen(4, 10);
			Neural* nptr = new Neural();
			nptr->Init();
			nptr->GenerateWeight();
			nptr->TrainSet(imageList, TRAINNUM, 0.001, 10000);
		}
		*/


    }
    catch(string errs)
    {
        cerr << errs << endl;
    }

    if(nn!= NULL)
        delete nn;

    cout << "Done" << endl;
	getchar();
    return 0;
}
