#include "MNeural.h"
#include <iostream>


using namespace std;

const int _NUMLAYER = 3;

MNeural::MNeural()
{
    this->units = new int[_NUMLAYER];
    this->units[0] = UNITS[0];
    this->units[1] = UNITS[1];
    this->units[2] = UNITS[NUM_LAYERS-1];


    this->eta = LEARNCOST;

    this->tarptr = new TBinGen(this->units[_NUMLAYER-1 ], OUTPUT);
}

MNeural::MNeural(int* units, float learnRate, TargetGenBase* tarptr)
{
    this->units = units;

    this->eta = learnRate;



    this->tarptr = tarptr;
}

MNeural::~MNeural()
{
    delete[] this->units;
}

void MNeural::Init(int numSample)
{
    this->numInput = this->units[0];
    this->numOutput = this->units[_NUMLAYER-1];
    this->numHidden = this->units[1];

    this->numSample = numSample;

    this->mInputValue.Resize(numInput, numSample);

    this->mI2HWeight.Resize(numHidden, numInput);


    this->mH2OWeight.Resize(numOutput, numHidden);


    this->mHideBias.Resize(numHidden, 1);


    this->mOutputBias.Resize(numOutput, 1);

}

void MNeural::GenerateWeight()
{
    this->mI2HWeight.RandomInitialize(HIGH, LOW);
    this->mH2OWeight.RandomInitialize(HIGH, LOW);
    this->mHideBias.RandomInitialize(HIGH, LOW);
    this->mOutputBias.RandomInitialize(HIGH, LOW);
}


void MNeural::TrainSet(Image* imageList, int count, float diff, int maxIter, float nStep, float maxStep)
{


    for(int i=0; i < count; i++)
    {
        int length = imageList[i].length;
        for(int j=0;j<length;j++)
        {
            this->mInputValue.m_pTMatrix (j, i) = imageList[i].content[j];
			
        }
		

    }

    Forward(this->mI2HWeight, this->mHideBias, this->mH2OWeight, this->mOutputBias);

    CMatrix mDemoOutput(this->numOutput, this->numSample);

    for(int i=0;i < count;i++)
    {
        float* target = tarptr->GetTarget(imageList[i].label);
        for(int j = 0;j<this->numOutput;j++)
        {
            mDemoOutput.m_pTMatrix (j, i) = target[j];
        }
    }

    CMatrix mOutputError(this->numOutput, this->numSample);
    mOutputError = mDemoOutput - this->mOutOutput;

	//cout << "mDemoOutput "<< endl;
	//mDemoOutput.Print();
	//cout << "mOutputError" << endl;
	//mOutputError.Print();


    float sysErrOld = mOutputError.GetSystemError();
	printf("error = %.3f\n", sysErrOld);
    for(int loopTime = 1; loopTime < maxIter; loopTime++)
    {
        printf("Iter %d\n", loopTime);
        if(sysErrOld < diff)
        {
            loopTime--;
            break;
        }

        CMatrix mExHideOutput;
        mExHideOutput.nncpyi(this->mHideOutput, this->numOutput);

        CMatrix mOutDelta(this->numOutput, this->numSample);
		
        mOutDelta = this->mOutOutput / this->mOutOutput - 1;

		//cout << "mOutDelta" << endl;
		//mOutDelta.Print();


        CMatrix mExOutputDelta;
        mExOutputDelta.nncpyd(mOutDelta);
		//cout << "mExOutputDelta" << endl;
		//mExOutputDelta.Print();

        CMatrix t_mH2OWeight(this->mH2OWeight.GetColCount(), this->mH2OWeight.GetRowCount());
        t_mH2OWeight = this->mH2OWeight.Transpose();

        CMatrix mExHideDelta;
        mExHideDelta = (1 - mExHideOutput / mExHideOutput) / (t_mH2OWeight * mExOutputDelta);

        CMatrix mExInput;
		//cout << "mInputValue" << endl;
		//this->mInputValue.Print();
        mExInput.nncpyi(this->mInputValue, this->numOutput);
		//cout << "mExINput" << endl;
		//mExInput.Print();

        CMatrix mJ11;
        mJ11.nncpy(mExHideDelta.Transpose(), mExInput.GetRowCount());

        CMatrix mJ12;
        mJ12.nncpyi(mExInput.Transpose(), mExHideDelta.GetRowCount());

        CMatrix mJ1;
        mJ1 = mJ11 / mJ12;

        CMatrix mJ21;
        mJ21.nncpy(mExOutputDelta.Transpose(), mExHideOutput.GetRowCount());

        CMatrix mJ22;
        mJ22.nncpyi(mExHideOutput.Transpose(), mExOutputDelta.GetRowCount());

        CMatrix mJ2;
        mJ2 = mJ21 / mJ22;

        int colArr[4];
        colArr[0] = mJ1.GetColCount();
        colArr[1] = colArr[0] + mExHideDelta.GetRowCount();
        colArr[2] = colArr[1] + mJ2.GetColCount();
        colArr[3] = colArr[2] + mExOutputDelta.GetRowCount();
		
		//cout << "mJ1" << endl;
		//mJ1.Print();

		//cout << "mExHideDelta" << endl;
		//mExHideDelta.Print();

		//cout << "mJ2" << endl;
		//mJ2.Print();

		//cout << "mExOutputDelta" << endl;
		//mExOutputDelta.Print();

        CMatrix matrixZ(this->numOutput * this->numSample, colArr[3]);
        mJ1.CopyTo(matrixZ, 0, 0);
        mExHideDelta.Transpose().CopyTo(matrixZ, 0, colArr[0]);
        mJ2.CopyTo(matrixZ, 0, colArr[1]);
        mExOutputDelta.Transpose().CopyTo(matrixZ, 0, colArr[2]);

		//cout << "matrixZ" << endl;
		//matrixZ.Print();

        CMatrix c_mOutputError;
        c_mOutputError = mOutputError.MergeColumnsToColumnVector();

        CMatrix mJE;
        mJE = matrixZ.Transpose() * c_mOutputError;



        printf("About to calculate J(x)T * J(x)\n");

        CMatrix mJJ;
        mJJ = matrixZ.Transpose() * matrixZ;

        printf("J(x)T * J(x) done\n");

		//cout << "mJJ" << endl;
		//mJJ.Print();

        // 定义新的输入层到隐含层的权值
		CMatrix mNewI2HWeight;
		// 定义的新的隐含层的阀值
		CMatrix mNewHideBias;
		// 定义新的隐含层到输出层的权值
		CMatrix mNewH2OWeight;
		// 定义新的输出层的阀值
		CMatrix mNewOutputBias;

		CMatrix newOutputError(this->numOutput, this->numSample);



        float sysErrNew;
		while(nStep <= maxStep)
		{
		    CMatrix matrixI(matrixZ.GetColCount(), matrixZ.GetColCount());
		    matrixI.Eye();

            #ifdef __DEBUG
            printf("About to calculate the formula\n");
            #endif

		    CMatrix matrixDX;

			CMatrix temp = mJJ + matrixI * nStep;
			printf("About to inverse\n");

			cout << "temp" << endl;
			temp.Print();

			CMatrix temp2 = temp.Inverse();


			

			printf("inverse completed\n");
		    matrixDX = temp2 * mJE * (-1.0);
			
			//cout << "matrixDX" << endl;
			//matrixDX.Print();

		    int nIndex = 0;
		    CMatrix mI2HWeightChange(this->numHidden, this->numInput);
		    matrixDX.CopySubMatrixFromVector(mI2HWeightChange, 0);
			


		    CMatrix mHideBiasChange(this->numHidden, 1);
		    matrixDX.CopySubMatrixFromVector(mHideBiasChange, colArr[0]);

		    CMatrix mH2OWeightChange(this->numOutput, this->numHidden);
		    matrixDX.CopySubMatrixFromVector(mH2OWeightChange, colArr[1]);

			//cout << "H2OWeight CHange" << endl;
			//mH2OWeightChange.Print();


		    CMatrix mOutputBiasChange(this->numOutput, 1);
		    matrixDX.CopySubMatrixFromVector(mOutputBiasChange, colArr[2]);
			
			//cout << "OutputBiasChange" << endl;
			//mOutputBiasChange.Print();


		    mNewI2HWeight = mI2HWeight + mI2HWeightChange;
		    mNewHideBias = mHideBias + mHideBiasChange;
		    mNewH2OWeight = mH2OWeight + mH2OWeightChange;
		    mNewOutputBias = mOutputBias + mOutputBiasChange;

			
		    Forward(mNewI2HWeight, mHideBias, mH2OWeight, mOutputBias);

            mOutputError = mDemoOutput - this->mOutOutput;

            sysErrNew = mOutputError.GetSystemError();
			printf("New error = %.3f, Old error = %.3f, nStep = %.5f\n", sysErrNew, sysErrOld, nStep);
            if(sysErrNew < sysErrOld)
            {
                break;
            }
            else
            {
                nStep *= 10;
            }

		}

		if(nStep > maxStep)
		{
		    loopTime--;
		    printf("Train failed\n");
		    return;
		}

		nStep *= 0.1;

		mI2HWeight = mNewI2HWeight;
		mHideBias = mNewHideBias;
		mH2OWeight = mNewH2OWeight;
		mOutputBias = mNewOutputBias;
		
		sysErrOld = sysErrNew;

		printf("Iter %d : Err = %.3f\n", loopTime, sysErrOld);
    }

}

void MNeural::Forward(CMatrix& _mI2HWeight, CMatrix& _mHideBias, CMatrix& _mH2OWeight, CMatrix& _mOutputBias)
{
    printf("Forward ...\n");
    CMatrix cMExHideBias;
    cMExHideBias.nncpyi(_mHideBias, this->numSample);
	//cout << "_mHideBias" << endl;
	//_mHideBias.Print();
	//cout << "cMExHideBias" << endl;
	//cMExHideBias.Print();


    CMatrix cMHidePureInput(this->numHidden, this->numSample);
// 	cout << "mI2HWeight" << endl;
// 	_mI2HWeight.Print();
	//cout << "mInputValue" << endl;
	//this->mInputValue.Print();
    cMHidePureInput = _mI2HWeight * this->mInputValue;
	//cout << "cMHidePureInput" << endl;
	//cMHidePureInput.Print();


    //cMHidePureInput += cMExHideBias;


	//cout << "cMHidePureInput" << endl;
	//cMHidePureInput.Print();
    //CMatrix cMHideOutput(this->numHidden, this->numSample);

    this->mHideOutput = cMHidePureInput.Sigmoid();
	//cout << "After Sigmoid()" << endl;
	//this->mHideOutput.Print();

    CMatrix cMExOutputBias;
	//cout << "OutputBias" << endl;
	//_mOutputBias.Print();
    cMExOutputBias.nncpyi(_mOutputBias, this->numSample);
	//cout << "cMExOutputBias" << endl;
	//cMExOutputBias.Print();

    CMatrix cMOutPureInput(this->numOutput, this->numSample);
	//cout << "_mH20Weight" << endl;
	//_mH2OWeight.Print();
	//cout << "mHideOutput" << endl;
	//this->mHideOutput.Print();

    cMOutPureInput = _mH2OWeight * this->mHideOutput;
	//cout << "cMOutPureInput" << endl;
	//cMOutPureInput.Print();

    //cMOutPureInput += cMExOutputBias;


	//cout << "cMOutPureInput" << endl;
	//cMOutPureInput.Print();
    this->mOutOutput = cMOutPureInput.Sigmoid();

	//cout << "mOutOutput" << endl;
	//this->mOutOutput.Print();
}

bool MNeural::Test(float* input, int label)
{
	this->numSample = 1;
	this->mInputValue.Resize(this->numInput, 1);
	
	for(int i=0;i<this->numInput;i++)
	{
		this->mInputValue.m_pTMatrix(i, 0) = input[i];
	}
    Forward(this->mI2HWeight, this->mHideBias, this->mH2OWeight, this->mOutputBias);

	float* output = new float[this->numOutput];
	for(int i=0;i<this->numOutput;i++)
	{
		output[i] = this->mOutOutput.m_pTMatrix(i, 0);
	}

	int predict = tarptr->Check(output);
	
	printf("%d -> %d\n", label, predict);
	delete[] output;
	return label == predict;
}

void MNeural::TestSet(Image* imageList, int count)
{
	int right = 0;
	for(int i=0;i<count;i++)
	{
		if(Test(imageList[i].content, imageList[i].label))
		{
			right++;
		}
	}
	printf("Correct percent = %.3f%%\n", ((float)(right*100))/count);
}
