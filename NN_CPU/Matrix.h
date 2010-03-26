#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

#include "global.h"


using namespace boost::numeric::ublas;
class CMatrix
{

	/************************************************************************
	*				the interface function of the class CMatrix 			*
	************************************************************************/
public:

	/////////////////////////////////////////////////////////////////////////
	// Construction and Destruction
    CMatrix();
	CMatrix(const CMatrix& cMatrixB);

	virtual ~CMatrix();

	matrix<VALTYPE> m_pTMatrix;				// ָ������ͷָ��

	/////////////////////////////////////////////////////////////////////////
	// According to the parameters nRow & nCol to construct a matrix
	CMatrix(int nRow, int nCol);


	/////////////////////////////////////////////////////////////////////////
	// This function initialize the matrix :
	//		the matrix which has been initialized has 0 row & 0 column, and
	// all elements in it is zeros.
	//
	void Initialize();

	/////////////////////////////////////////////////////////////////////////
	// This function initialize the matrix :
	//		the matrix which has been initialized has 0 row & 0 column, and
	// all elements in it is zeros.
	//
	void InitializeZero();

	/////////////////////////////////////////////////////////////////////////
	// To make random in the elements of the matrix and the elements of the
	// matrix has been randomized between -1 and 1.These elements must be
	// decimal fractions.
	//
	void RandomInitialize(float, float);


    void Resize(int, int);
	/////////////////////////////////////////////////////////////////////////
	// Overload Operations

	// 'CMatrix + CMatrix'
	CMatrix operator + (const CMatrix& cMatrixB);
	// 'CMatrix - CMatrix'
	CMatrix operator - (const CMatrix& cMatrixB);

	CMatrix operator - (double nValue);

	// 'CMatrix * CMatrix'
	CMatrix operator * (const CMatrix& cMatrixB);



	// 'CMatrix * double'
	CMatrix operator * (double nValue);
	// 'CMatrix = CMatrix'
	CMatrix& operator = (const CMatrix& cMatrixB);



	// 'CMatrix += CMatrix'
	CMatrix& operator += (CMatrix& cMatrixB);
	// 'CMatrix .* CMatrix'
	CMatrix operator / (const CMatrix& cMatrixB);


	/////////////////////////////////////////////////////////////////////////
	// Transpose the matrix
	//
	CMatrix Transpose();

	/////////////////////////////////////////////////////////////////////////
	// Inverse the matrix
	//
	CMatrix Inverse();



	/////////////////////////////////////////////////////////////////////////
	// Get the matrix Row Number
	//
	int GetRowCount() const
	{
		return m_nRow;
	}

	/////////////////////////////////////////////////////////////////////////
	// Get the matrix Colum Number
	//
	int GetColCount() const
	{
		return m_nCol;
	}




	/////////////////////////////////////////////////////////////////////////
	//	Copy data from a vector to a matrix
	//	Parameter:
	//		[out]	cMatrix ----> the returned value
	//		[in]	nIndex	----> the index in vector
	//	Notes:
	//		the object copied must be vector!!!
	//
	void CopySubMatrixFromVector(CMatrix& cMatrix,int nIndex);

	/////////////////////////////////////////////////////////////////////////
	// Copy data from sub matrix to another matrix
	// Parameter:
	//		[out]	cMatrix ----> ������Ӿ��󷵻صĽ��
	//		[in]	nStartX ----> �Ӿ����ھ����е���ʼ����,��Ӧ��,������1��ʼ
	//		[in]	nStartY ----> �Ӿ����ھ����е���ʼ����,��Ӧ��,������1��ʼ
	//
	void CopySubMatrix(CMatrix& cMatrix,int nStartX,int nStartY);

	/////////////////////////////////////////////////////////////////////////
	// Copy Matrix
	//	Parameter:
	//		[in]	cMatrix ----> be copied matrix
	//
	void CopyMatrix(CMatrix cMatrix);

	/////////////////////////////////////////////////////////////////////////
	// ����������е�Ԫ�ذ��кϳ�һ��
	//	���磺
	//		matrix = [
	//			1	2	3
	//			4	5	6
	//			7	8	9
	//				]
	//		CMatrix cMatrix = matrix.MergeColumnsToColumnVector();
	//		cMatrix =
	//			[	1
	//				4
	//				7
	//				2
	//				5
	//				8
	//				3
	//				6
	//				9	]
	//
	CMatrix MergeColumnsToColumnVector();

	/////////////////////////////////////////////////////////////////////////
	// �Ծ��������е�Ԫ�ؽ���һ�η����Ա任:
	//		�任���ֵy��任ǰ��ֵ�Ĺ�ϵ��:
	//			y = f(x) = 1 / (1 + exp(-x))	( 0 < f(x) < 1)
	//
	CMatrix Sigmoid();

	/////////////////////////////////////////////////////////////////////////
	// �Ծ��������е�Ԫ�ؽ���һ�η����Ա任:
	//		�任���ֵy��任ǰ��ֵ�Ĺ�ϵ��:
	//			y = f'(x) = (1 / (1 + exp(-x)))'	( 0 < f(x) < 1)
	//			  = exp(-x)/((1 + exp(-x))*(1 + exp(-x)))
	//
	CMatrix SigmoidDerivative();


	/////////////////////////////////////////////////////////////////////////
	// �Ծ��������е�Ԫ�ؽ���һ�η����Ա任:
	//		�任���ֵy��任ǰ��ֵ�Ĺ�ϵ��:
	//			y = tanh(x) = (1 - exp(-x)) / (1 + exp(-x))
	//					 =  1 - 2 * exp(-x) / (1 + exp(-x))	( -1 < f(x) < 1)
	//
	CMatrix tanh();

	/////////////////////////////////////////////////////////////////////////
	// �Ծ��������е�Ԫ�ؽ���һ�η����Ա任:
	//		�任���ֵy��任ǰ��ֵ�Ĺ�ϵ��:
	//			y = tanh'(x) = ((1 - exp(-x)) / (1 + exp(-x)))'	( -1 < f(x) < 1)
	//					 = 	2*exp(-x)/((1 + exp(-x))*(1 + exp(-x)))
	//
	CMatrix tanhDerivative();

	/////////////////////////////////////////////////////////////////////////
	// �Ծ��������е�Ԫ�ؽ���һ�η����Ա任:
	//		�任���ֵy��任ǰ��ֵ�Ĺ�ϵ��:
	//			y = Tansig(x) = 2 / (1 + exp(-2 * x)) -1
	//
	CMatrix Tansig();

	/////////////////////////////////////////////////////////////////////////
	// �Ծ��������е�Ԫ�ؽ���һ�η����Ա任:
	//		�任���ֵy��任ǰ��ֵ�Ĺ�ϵ��:
	//			y = Tansig'(x) = (2 / (1 + exp(-2 * x)) -1)'
	//				= (2 / (1 + exp(-2 * x)) -1) * (2 / (1 + exp(-2 * x)) -1) -1
	//
	CMatrix TansigDerivative();


	/////////////////////////////////////////////////////////////////////////
	// �Ծ����е�Ԫ�ؽ���һ�β���:
	//		ʹ�����Ϊ��λ��
	//
	void Eye();

	/////////////////////////////////////////////////////////////////////////
	// Get System Error
	//
	double	GetSystemError() const;

	/////////////////////////////////////////////////////////////////////////
	// Make all the matrix elements to be changed into absolute value
	//
	CMatrix AbsoluteValue();

	/////////////////////////////////////////////////////////////////////////
	// Parameter:
	//		CMatrix& cMatrix:		������������Դ
	//		int nIndex:	�������������ڶ����еĿ�ʼ����λ��
	// Purpose:
	//		This function will copy all the data of the cMatrix
	// Notes:
	//		The object must be column vector!!!
	//
	void GetMatrixData(CMatrix& cMatrix, int nIndex);

	/////////////////////////////////////////////////////////////////////////
	// Parameter:
	//		CMatrix& cMatrix:		�����ľ���
	//		int nIndex:	�������������ڶ����еĿ�ʼ����λ��
	// Purpose:
	//		This function will copy part of the object data into cMatrix
	// Notes:
	//		The object must be column vector!!!
	//
	void SetMatrixData(CMatrix& cMatrix, int nIndex);

	/////////////////////////////////////////////////////////////////////////
	// Parameter:
	//		CMatrix& cMatrix:		������������Դ
	//		int nIndex:	�������������ڶ����еĿ�ʼ����λ��
	//		int nRow:		�������������ڱ����������е�������(��0��ʼ)
	// Purpose:
	//		This function will copy all the data of the cMatrix
	// Notes:
	//		The object must be column vector!!!
	//
	void GetMatrixRowData(CMatrix& cMatrix, int nIndex, int nRow);

	/////////////////////////////////////////////////////////////////////////
	// Parameter:
	//		CMatrix& cMatrix:		�����ľ���
	//		int nIndex:	�������������ڶ����еĿ�ʼ����λ��
	//		int nRow:		�����������ڱ��������е�������
	// Purpose:
	//		This function will copy part of the object data to fill the special
	// row of the cMatrix
	//	Notes:
	//		The object must be column vector!!!
	//
	void SetMatrixRowData(CMatrix& cMatrix, int nIndex, int nRow);

	/////////////////////////////////////////////////////////////////////////
	// Get the total value of the matrix
	double GetTotalElementValue();

	/////////////////////////////////////////////////////////////////////////
	// �Ծ��������չ
	//	ʵ�ֹ���:
	//		�Ծ��������������չ,nTimes��ÿ����չ�Ĵ���
	//
	void nncpyi(const CMatrix &cMatrix, int nTimes);

	/////////////////////////////////////////////////////////////////////////
	// �Ծ��������չ
	//	ʵ�ֹ���:
	//		�Ծ��������������չ
	//	matrix =	[
	//			1	2	3
	//			4	5	6
	//			7	8	9
	//				]
	//
	//		nncpyd(matrix)	=	[
	//			1	0	0	2	0	0	3	0	0
	//			0	4	0	0	5	0	0	6	0
	//			0	0	7	0	0	8	0	0	9
	//							]
	void nncpyd(CMatrix &cMatrix);

	/////////////////////////////////////////////////////////////////////////
	// �Ծ��������չ
	//	ʵ�ֹ���:
	//		�Ծ��������������չ,nTimes��ÿ����չ�Ĵ���
	//	matrix =	[
	//			1	2	3
	//			4	5	6
	//			7	8	9
	//				]
	//		nTimes = 2
	//
	//		nncpyd(matrix)	=	[
	//					1	2	3	1	2	3
	//					4	5	6	4	5	6
	//					7	8	9	7	8	9
	//							]
	//
	void nncpy (const CMatrix& cMatrix, int nTimes);


    void CopyTo(CMatrix& matrix, int startRow, int startCol);

	void Print();

private:

	int m_nRow;			// ������ӵ�е�����
	int m_nCol;			// ������ӵ�е�����


	/////////////////////////////////////////////////////////////////////////
	// ע��:
	//		���������þ����������������,�����е�Ԫ�ر����³�ʼ��Ϊ0
	/////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////
	// ���þ��������
	//
	void SetMatrixRowNumber(int nRow);

	/////////////////////////////////////////////////////////////////////////
	// ���þ��������
	//
	void SetMatrixColNumber(int nCol);

	/////////////////////////////////////////////////////////////////////////
	// ���þ����������
	//
	void SetMatrixRowAndCol(int nRow,int nCol);


	/////////////////////////////////////////////////////////////////////////
	// �������������
	//
	void SwapMatrixRow(int nRow1,int nRow2);

	/////////////////////////////////////////////////////////////////////////
	// �������������
	//
	void SwapMatrixCol(int nCol1,int nCol2);




};
CMatrix operator - (double nValue, const CMatrix& cMatrixB);
#endif // MATRIX_H_INCLUDED
