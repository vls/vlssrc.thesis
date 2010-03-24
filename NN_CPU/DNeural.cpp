#include "Neural.h"
#include <iostream>

using namespace std;

const float ALPHA = 0.5;

void DNeural::InitLayer()
{
    cout << "Init DLayer" << endl;
    for(int i = 0;i < NUM_LAYERS; i++)
    {
        int lowerUnits = i == 0 ? 0 : this->layers[i-1]->units;

        this->layers[i] = new DLayer(UNITS[i], lowerUnits);
    }

}

void DNeural::AdjustWeight()
{
	for(int l = 1; l < this->num_layer; l++)
	{
		for(int i = 0; i<this->layers[l]->units; i++)
		{
		    float Err = this->layers[l]->error[i];
			for(int j = 0; j < this->layers[l-1]->units; j++)
			{
				float Out = this->layers[l-1]->output[j];

                float dWeight = ((DLayer*)this->layers[l])->dWeight[i][j];

                float delta = this->learnRate * Err * Out + ::ALPHA * dWeight;


				this->layers[l]->weight[i][j] += delta;
				((DLayer*)this->layers[l])->dWeight[i][j] = this->learnRate * Err * Out;
				int zz= 0;
			}
			float delta = this->learnRate * Err;
			this->layers[l]->bias[i] += delta;

			int zz= 0;
		}
	}
}
