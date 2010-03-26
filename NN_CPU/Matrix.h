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

	matrix<VALTYPE> m_pTMatrix;				// 指向矩阵的头指针

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
	//		[out]	cMatrix ----> 矩阵的子矩阵返回的结果
	//		[in]	nStartX ----> 子矩阵在矩阵中的起始坐标,对应行,索引从1开始
	//		[in]	nStartY ----> 子矩阵在矩阵中的起始坐标,对应列,索引从1开始
	//
	void CopySubMatrix(CMatrix& cMatrix,int nStartX,int nStartY);

	/////////////////////////////////////////////////////////////////////////
	// Copy Matrix
	//	Parameter:
	//		[in]	cMatrix ----> be copied matrix
	//
	void CopyMatrix(CMatrix cMatrix);

	/////////////////////////////////////////////////////////////////////////
	// 将矩阵的所有的元素按列合成一列
	//	例如：
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
	// 对矩阵中所有的元素进行一次非线性变换:
	//		变换后的值y与变换前的值的关系是:
	//			y = f(x) = 1 / (1 + exp(-x))	( 0 < f(x) < 1)
	//
	CMatrix Sigmoid();

	/////////////////////////////////////////////////////////////////////////
	// 对矩阵中所有的元素进行一次非线性变换:
	//		变换后的值y与变换前的值的关系是:
	//			y = f'(x) = (1 / (1 + exp(-x)))'	( 0 < f(x) < 1)
	//			  = exp(-x)/((1 + exp(-x))*(1 + exp(-x)))
	//
	CMatrix SigmoidDerivative();


	/////////////////////////////////////////////////////////////////////////
	// 对矩阵中所有的元素进行一次非线性变换:
	//		变换后的值y与变换前的值的关系是:
	//			y = tanh(x) = (1 - exp(-x)) / (1 + exp(-x))
	//					 =  1 - 2 * exp(-x) / (1 + exp(-x))	( -1 < f(x) < 1)
	//
	CMatrix tanh();

	/////////////////////////////////////////////////////////////////////////
	// 对矩阵中所有的元素进行一次非线性变换:
	//		变换后的值y与变换前的值的关系是:
	//			y = tanh'(x) = ((1 - exp(-x)) / (1 + exp(-x)))'	( -1 < f(x) < 1)
	//					 = 	2*exp(-x)/((1 + exp(-x))*(1 + exp(-x)))
	//
	CMatrix tanhDerivative();

	/////////////////////////////////////////////////////////////////////////
	// 对矩阵中所有的元素进行一次非线性变换:
	//		变换后的值y与变换前的值的关系是:
	//			y = Tansig(x) = 2 / (1 + exp(-2 * x)) -1
	//
	CMatrix Tansig();

	/////////////////////////////////////////////////////////////////////////
	// 对矩阵中所有的元素进行一次非线性变换:
	//		变换后的值y与变换前的值的关系是:
	//			y = Tansig'(x) = (2 / (1 + exp(-2 * x)) -1)'
	//				= (2 / (1 + exp(-2 * x)) -1) * (2 / (1 + exp(-2 * x)) -1) -1
	//
	CMatrix TansigDerivative();


	/////////////////////////////////////////////////////////////////////////
	// 对矩阵中的元素进行一次操作:
	//		使矩阵变为单位阵
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
	//		CMatrix& cMatrix:		被拷贝的数据源
	//		int nIndex:	被拷贝的数据在对象中的开始索引位置
	// Purpose:
	//		This function will copy all the data of the cMatrix
	// Notes:
	//		The object must be column vector!!!
	//
	void GetMatrixData(CMatrix& cMatrix, int nIndex);

	/////////////////////////////////////////////////////////////////////////
	// Parameter:
	//		CMatrix& cMatrix:		被填充的矩阵
	//		int nIndex:	被拷贝的数据在对象中的开始索引位置
	// Purpose:
	//		This function will copy part of the object data into cMatrix
	// Notes:
	//		The object must be column vector!!!
	//
	void SetMatrixData(CMatrix& cMatrix, int nIndex);

	/////////////////////////////////////////////////////////////////////////
	// Parameter:
	//		CMatrix& cMatrix:		被拷贝的数据源
	//		int nIndex:	被拷贝的数据在对象中的开始索引位置
	//		int nRow:		被拷贝的数据在被拷贝对象中的行索引(从0开始)
	// Purpose:
	//		This function will copy all the data of the cMatrix
	// Notes:
	//		The object must be column vector!!!
	//
	void GetMatrixRowData(CMatrix& cMatrix, int nIndex, int nRow);

	/////////////////////////////////////////////////////////////////////////
	// Parameter:
	//		CMatrix& cMatrix:		被填充的矩阵
	//		int nIndex:	被拷贝的数据在对象中的开始索引位置
	//		int nRow:		被填充的数据在被填充对象中的行索引
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
	// 对矩阵进行拓展
	//	实现功能:
	//		对矩阵的列数进行拓展,nTimes是每列拓展的次数
	//
	void nncpyi(const CMatrix &cMatrix, int nTimes);

	/////////////////////////////////////////////////////////////////////////
	// 对矩阵进行拓展
	//	实现功能:
	//		对矩阵的列数进行拓展
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
	// 对矩阵进行拓展
	//	实现功能:
	//		对矩阵的列数进行拓展,nTimes是每列拓展的次数
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

	int m_nRow;			// 矩阵所拥有的行数
	int m_nCol;			// 矩阵所拥有的列数


	/////////////////////////////////////////////////////////////////////////
	// 注意:
	//		在重新设置矩阵的行数和列数后,矩阵中的元素被重新初始化为0
	/////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////
	// 设置矩阵的行数
	//
	void SetMatrixRowNumber(int nRow);

	/////////////////////////////////////////////////////////////////////////
	// 设置矩阵的列数
	//
	void SetMatrixColNumber(int nCol);

	/////////////////////////////////////////////////////////////////////////
	// 设置矩阵的行列数
	//
	void SetMatrixRowAndCol(int nRow,int nCol);


	/////////////////////////////////////////////////////////////////////////
	// 交换矩阵的两行
	//
	void SwapMatrixRow(int nRow1,int nRow2);

	/////////////////////////////////////////////////////////////////////////
	// 交换矩阵的两列
	//
	void SwapMatrixCol(int nCol1,int nCol2);




};
CMatrix operator - (double nValue, const CMatrix& cMatrixB);
#endif // MATRIX_H_INCLUDED
