/////////////////////////////////////////////////////////////////////////////
// Matrix.cpp : Implementation of the class Matrix
//
/////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "Matrix.h"
#include <math.h>
#include <stdlib.h>

#include <conio.h>
#include <stdio.h>


using namespace std;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////


CMatrix::CMatrix() : m_pTMatrix(0, 0)
{
    m_nRow = 0;
    m_nCol = 0;


}

CMatrix::~CMatrix()
{

}


CMatrix::CMatrix(int nRow,int nCol) : m_pTMatrix(nRow, nCol)
{

	for(int i=0; i < nRow; i++)
	{
		for(int j=0; j < nCol; j++)
		{
			m_pTMatrix(i, j) = 0.0f;
		}
	}

	// 对对象变量赋值
	m_nRow	= nRow;
	m_nCol	= nCol;

}


CMatrix::CMatrix(const CMatrix& cMatrixB) : m_pTMatrix(cMatrixB.GetRowCount(), cMatrixB.GetColCount())
{
	// Initialize the variable
	m_nRow = cMatrixB.m_nRow ;
	m_nCol = cMatrixB.m_nCol ;
	m_pTMatrix = cMatrixB.m_pTMatrix ;

}


/////////////////////////////////////////////////////////////////////////////
// CMatrix member functions
//

CMatrix CMatrix::operator +(const CMatrix& cMatrixB)
{
	// 要满足矩阵相加的条件: 行列数目相等!
	if(m_nRow != cMatrixB.m_nRow || m_nCol != cMatrixB.m_nCol )
	{
		throw string("执行相加的两个矩阵维数不相等!");
	}

	CMatrix	cMatrix(m_nRow, m_nCol);

	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < m_nCol; j++)
		{
			cMatrix.m_pTMatrix (i, j) = m_pTMatrix (i, j) + cMatrixB.m_pTMatrix(i, j);
		}
	}

	return	cMatrix;

}


CMatrix CMatrix::operator -(const CMatrix& cMatrixB)
{
	// 要满足矩阵相加的条件: 行列数数目相等!
	if(m_nRow != cMatrixB.m_nRow || m_nCol != cMatrixB.m_nCol )
	{
		throw string("执行相减的两个矩阵维数不相等!");
	}

	CMatrix cMatrix(m_nRow, m_nCol);

	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < m_nCol; j++)
		{
			cMatrix.m_pTMatrix (i, j) = m_pTMatrix (i, j) - cMatrixB.m_pTMatrix(i, j);
		}
	}

	return	cMatrix;

}

CMatrix CMatrix::operator - (double nValue)
{
	CMatrix	cMatrix(m_nRow, m_nCol);

	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < m_nCol; j++)
		{
			cMatrix.m_pTMatrix (i, j) = m_pTMatrix(i, j) - nValue;
		}
	}

	return cMatrix;
}


CMatrix CMatrix::operator *(const CMatrix& cMatrixB)
{
	if( m_nCol != cMatrixB.m_nRow )
	{
		throw string("执行相乘的两个矩阵维数不满足相乘的条件!");
	}

	CMatrix cResultMatrix(m_nRow,cMatrixB.m_nCol);

	for(int i=0; i < m_nRow; i++)
	{
	    //printf("Row = %d/%d\n", i, m_nRow);
		for(int j=0; j < cMatrixB.m_nCol; j++)
		{

            for(int m=0; m < m_nCol; m++)
			{
				cResultMatrix.m_pTMatrix (i, j) +=  m_pTMatrix (i, m) * cMatrixB.m_pTMatrix(m, j);
			}
		}
	}

	return cResultMatrix;
}


CMatrix CMatrix::operator * (double nValue)
{
	CMatrix cMatrix(m_nRow, m_nCol);

	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < m_nCol; j++)
		{
			cMatrix.m_pTMatrix (i, j) =m_pTMatrix(i, j) * nValue;
		}
	}

	return cMatrix;
}


CMatrix& CMatrix::operator =(const CMatrix& cMatrixB)
{
    /*
	if( (m_nRow != cMatrixB.m_nRow) || (m_nCol != cMatrixB.m_nCol) )
	{
		throw string("等号左右两边的矩阵的维数不相等!");
		return *this;	// return invalid value
	}*/
    m_pTMatrix.resize (cMatrixB.m_nRow, cMatrixB.m_nCol);
	// 给变量赋值
	m_nRow = cMatrixB.m_nRow ;
	m_nCol = cMatrixB.m_nCol ;
	m_pTMatrix = cMatrixB.m_pTMatrix ;


	return *this;
}

CMatrix& CMatrix::operator += (CMatrix& cMatrixB)
{
	if(m_nRow != cMatrixB.m_nRow || m_nCol != cMatrixB.m_nCol )
	{
		//printf("错误!执行相加的两个矩阵维数不相等!\n");
		throw string("运算符的两边矩阵的维数不相等!");
		return *this;	// return invalid value
	}

	// 赋值操作
	for(int i=0; i < cMatrixB.m_nRow; i++)
	{
		for(int j=0; j< cMatrixB.m_nCol; j++)
		{
			m_pTMatrix (i, j) += cMatrixB.m_pTMatrix(i, j);
		}
	}

	return *this;

}


CMatrix CMatrix::Transpose()
{
	CMatrix cMatrix(m_nCol,m_nRow);

	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < m_nCol; j++)
		{
			cMatrix.m_pTMatrix (j, i) = m_pTMatrix(i, j);
		}
	}

	return cMatrix;
}

/////////////////////////////////////////////////////////////////////////////
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
/////////////////////////////////////////////////////////////////////////////

CMatrix CMatrix::MergeColumnsToColumnVector()
{
	CMatrix cMatrix(m_nRow * m_nCol,(int)1);

	// 对矩阵赋值
	for(int j=0; j < m_nCol; j++)
	{
		for(int i=0; i < m_nRow; i++)
		{
			cMatrix.m_pTMatrix (i + j * m_nRow, (int)0) = m_pTMatrix(i, j);
		}
	}

	return cMatrix;

}

/////////////////////////////////////////////////////////////////////////////
// Get the total value of the matrix
/////////////////////////////////////////////////////////////////////////////

double CMatrix::GetTotalElementValue()
{
	double	nTotalValue = 0;

	for(int i=0; i < m_nRow; i++)
	{
		for( int j=0; j < m_nCol; j++)
		{
			nTotalValue += m_pTMatrix (i, j);
		}
	}

	return nTotalValue;
}

/////////////////////////////////////////////////////////////////////////////
// Get System Error
/////////////////////////////////////////////////////////////////////////////

double	CMatrix::GetSystemError() const
{
	double	nSystemError = 0;

	for(int i=0; i < m_nRow; i++)
	{
		for( int j=0; j < m_nCol; j++)
		{
			nSystemError += m_pTMatrix (i, j) * m_pTMatrix(i, j);
		}
	}

	return nSystemError;

}

/////////////////////////////////////////////////////////////////////////////
// Make all the matrix elements to be changed into absolute value
/////////////////////////////////////////////////////////////////////////////

CMatrix CMatrix::AbsoluteValue ()
{
	CMatrix cMatrix(m_nRow, m_nCol);

	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < m_nCol; j++)
		{
			cMatrix.m_pTMatrix (i, j) = fabs( m_pTMatrix(i, j));

		}

	}

	return cMatrix;

}

/*
CMatrix CMatrix::Inverse()
{
	/////////////////////////////////////////////////////////////////////////
	// Using Gauss - Jordan Method
	// 参考书目: 计算机数值方法 --->施吉林 陈桂枝
	/////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////
	// 判断是否是可逆阵:
	//		可逆阵一定是方阵!!!

	if ( m_nRow != m_nCol)
	{
		//printf("错误!矩阵的行列数不相等,是非可逆阵!\n");
		throw string("矩阵的行列数不相等,是非可逆阵!");
	}

	CMatrix cMatrix(m_nRow, m_nCol);

	//***********************************************************************
	// 思路:(非常规思维!)
	//		动态分配整型数组(2*m_nCol)来存储每次交换的行坐标的值
	//		不论有没有行交换都记录在数组中,
	//		1.没进行行交换的两个数据相等,在SwapMatrixRow()函数中
	//		检测到两个值相等立即返回,在SwapMatrixCol()函数中也一样,
	//		检测到两个值相等立即返回,不占用系统资源;
	//		2.不相等的就交换
	//***********************************************************************

	//	分配内存
	int *pIntArray = new int [2*m_nCol];

	// nSetp --- 约化步数,按列展开
	for(int k=0; k < cMatrix.m_nCol; k++)
	{
		/////////////////////////////////////////////////////////////////////
		// 进行行交换 ---> 游戏规则:
		// 为保证计算过程的数值稳定性,在第k步约化时,先在{a(ik)}|i=k->n中选按
		// 模最大者作为约化主元素,并交换矩阵相应的行

		// 标记主元素
		double nMaxElement = cMatrix.m_pTMatrix(k, k);
		// 标记主元素所在的行数
		int nMainRow = k;

		for(int nCount = k+1; nCount < cMatrix.m_nCol; nCount++)
		{
			if( fabs(nMaxElement) < fabs(cMatrix.m_pTMatrix(nCount, k)) )
			{
				nMaxElement = cMatrix.m_pTMatrix (nCount, k);
				nMainRow = nCount;
			}
		}

		// 将欲交换的行数存在数组中
		pIntArray [2*k] = k;
		pIntArray [2*k+1] = nMainRow;


		// 交换行
		cMatrix.SwapMatrixRow(k,nMainRow);

		//Display();

		//	判断是否是可逆阵
		if(cMatrix.m_pTMatrix (k, k) == 0)
		{
			//printf("错误!此矩阵为非可逆阵!\n");
			throw string("此矩阵为非可逆阵,没有逆矩阵!");
		}

		cMatrix.m_pTMatrix (k, k) = 1/(cMatrix.m_pTMatrix(k, k));


		// 算主列
		for(int i=0; i < cMatrix.m_nRow; i++)
		{
			if( i != k)
				cMatrix.m_pTMatrix (i, k) = -(cMatrix.m_pTMatrix (k, k)) * (cMatrix.m_pTMatrix(i, k));

			//int nTempValue = m_pTMatrix(i, k);

		}

		//printf("\n");

		// 约化非主行
		for(int m=0; m < cMatrix.m_nRow; m++)
		{
			if ( m == k)
				continue;

			for(int n=0; n < cMatrix.m_nCol; n++)
			{
				if ( n == k)
					continue;

				cMatrix.m_pTMatrix (m, n) += cMatrix.m_pTMatrix (m, k) * cMatrix.m_pTMatrix(k, n);

				//printf("%10f ",m_pTMatrix (m, n));

			}

			//printf("\n");

		}

		// 算主行
		for(int j=0; j < cMatrix.m_nCol; j++)
		{
			if( j != k)
				cMatrix.m_pTMatrix (k, j) = (cMatrix.m_pTMatrix (k, k)) * (cMatrix.m_pTMatrix (k, j));

		}

	}


	/////////////////////////////////////////////////////////////////////////
	// 进行列交换 ---> 对交换行后的矩阵进行列交换 ---> 还原矩阵
	// 游戏规则:
	// 将开始矩阵中进行的行交换 ---> 现用相对应的列交换进行还原,即可得到所求的
	// 逆矩阵

	for(int i=2*m_nCol-1; i > 0; i--)
	{
		cMatrix.SwapMatrixCol(pIntArray[i],pIntArray[i-1]);
		i--;
	}

	delete []pIntArray;

	return cMatrix;

}
*/

CMatrix CMatrix::Inverse()
{
	using namespace boost::numeric::ublas;
	typedef permutation_matrix<std::size_t> pmatrix;
	

	CMatrix inverse(m_nRow, m_nCol);

	// create a working copy of the input
	matrix<VALTYPE> A(this->m_pTMatrix);
	// create a permutation matrix for the LU-factorization
	pmatrix pm(A.size1());


	// perform LU-factorization
	int res = lu_factorize(A, pm);
	if( res != 0 )
	{
		throw string("无法求出逆矩阵");
	}


	// create identity matrix of "inverse"
	inverse.m_pTMatrix.assign(identity_matrix<VALTYPE>(A.size1()));


	// backsubstitute to get the inverse
	lu_substitute(A, pm, inverse.m_pTMatrix);


	return inverse;
}

void CMatrix::SwapMatrixRow(int nRow1,int nRow2)
{
	if( nRow1 == nRow2)
		return;

	double *pArray = new double;

	for(int i=0; i < m_nCol; i++)
	{
		// Swap the datum of the two rows
		pArray[0] = m_pTMatrix (nRow1, i);
		m_pTMatrix (nRow1, i) = m_pTMatrix (nRow2, i);
		m_pTMatrix (nRow2, i) = pArray[0];
	}

	delete pArray;
}


void CMatrix::SwapMatrixCol(int nCol1,int nCol2)
{
	if( nCol1 == nCol2)
		return;

	double *pArray = new double;
	for(int i=0; i < m_nRow; i++)
	{
		// Swap the datum of the two columns
		pArray[0] = m_pTMatrix (i, nCol1);
		m_pTMatrix (i, nCol1) = m_pTMatrix (i, nCol2);
		m_pTMatrix (i, nCol2) = pArray[0];
	}

	delete pArray;
}




/////////////////////////////////////////////////////////////////////////////
// 对矩阵中的元素进行一次操作:
//		使矩阵变为单位阵
/////////////////////////////////////////////////////////////////////////////

void CMatrix::Eye()
{
	// Verify whether the rows is equal to the columns or not
	if(m_nRow != m_nCol)
	{
		throw string("此矩阵的行列数不相等!不能转变为单位阵!");
		return;
	}

	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < m_nCol; j++)
		{
			if(i == j)
			{
				m_pTMatrix (i, j) =	1;
			}
			else
			{
				m_pTMatrix (i, j) =	0;
			}
		}

	}


}


/////////////////////////////////////////////////////////////////////////////
// Parameter:
//		CMatrix& cMatrix:		被拷贝的数据源
//		int nIndex:	被拷贝的数据在对象中的开始索引位置,从0开始
// Purpose:
//		This function will copy all the data of the cMatrix
// Notes:
//		此对象必须是列向量!!!
/////////////////////////////////////////////////////////////////////////////

void CMatrix::GetMatrixData(CMatrix& cMatrix, int nIndex)
{
	if(m_nCol != 1)
	{
		throw string("拷贝的矩阵不是列向量!");
		return;
	}

	if((m_nRow - nIndex) < (cMatrix.m_nRow * cMatrix.m_nCol))
	{
		throw string("拷贝矩阵的空间容量不足!");
		return;
	}

	for(int i=0; i < cMatrix.m_nRow; i++)
	{
		for(int j=0; j < cMatrix.m_nCol; j++)
		{
			m_pTMatrix(nIndex + i * cMatrix.m_nCol + j, 0) = cMatrix.m_pTMatrix (i, j);
		}

	}

}


/////////////////////////////////////////////////////////////////////////////
// Parameter:
//		CMatrix& cMatrix:		被填充的矩阵
//		int nIndex:	被拷贝的数据在对象中的开始索引位置
// Purpose:
//		This function will copy part of the object data into cMatrix
// Notes:
//		The object must be column vector!!!
/////////////////////////////////////////////////////////////////////////////

void CMatrix::SetMatrixData(CMatrix& cMatrix, int nIndex)
{
	// Verify whether the colunm number is 1
	if(m_nCol != 1)
	{
		throw string("本矩阵对象不是列向量,不满足条件!");
		return;
	}

	// Verify whether the number of the object element is enough to be copyed
	if((m_nRow - nIndex) < (cMatrix.m_nRow * cMatrix.m_nCol))
	{
		throw string("对象中的元素数量不足!");
		return;
	}


	for(int i=0; i < cMatrix.m_nRow; i++)
	{
		for(int j=0; j < cMatrix.m_nCol; j++)
		{
			cMatrix.m_pTMatrix (i, j) = m_pTMatrix (nIndex + i * cMatrix.m_nCol + j, 0);

			// Using for debugging
			//int nIndexNumber = nIndex + i * cMatrix.m_nRow + j;
			//double nData = cMatrix.m_pTMatrix (i, j);

		}
	}

}


/////////////////////////////////////////////////////////////////////////////
// Parameter:
//		CMatrix& cMatrix:		被填充的矩阵
//		int nIndex:	被拷贝的数据在对象中的开始索引位置
//		int nRow:		被填充的数据在被填充对象中的行索引
// Purpose:
//		This function will copy part of the object data to fill the special
// row of the cMatrix
//	Notes:
//		The object must be column vector!!!
/////////////////////////////////////////////////////////////////////////////

void CMatrix::SetMatrixRowData(CMatrix& cMatrix, int nIndex, int nRow)
{
	// Verify whether the column number is 1
	if(m_nCol != 1)
	{
		throw string("本矩阵对象不是列向量,不满足条件!");
		return;
	}

	// Verify whether the number of the object element is enough to be copyed
	if((m_nRow - nIndex) < cMatrix.m_nCol )
	{
		throw string("对象的元素数量不足!");
		return;
	}

	for(int i=0; i < cMatrix.m_nCol; i++)
	{
		cMatrix.m_pTMatrix (nRow, i) = m_pTMatrix (nIndex + i, (int)0);
	}

}


/////////////////////////////////////////////////////////////////////////////
// Parameter:
//		CMatrix& cMatrix:		被拷贝的数据源
//		int nIndex:	被拷贝的数据在对象中的开始索引位置
//		int nRow:		被拷贝的数据在被拷贝对象中的行索引(从0开始)
// Purpose:
//		This function will copy all the data of the cMatrix
//	Notes:
//		此对象必须是列向量!!!
/////////////////////////////////////////////////////////////////////////////

void CMatrix::GetMatrixRowData(CMatrix& cMatrix, int nIndex, int nRow)
{
	if(m_nCol != 1)
	{
		throw string("拷贝的矩阵不是列向量!");
		return;
	}

	if((m_nRow - nIndex) < cMatrix.m_nCol)
	{
		throw string("拷贝矩阵的空间容量不足!");
		return;
	}

	for(int i=0; i < cMatrix.m_nCol; i++)
	{
		m_pTMatrix (nIndex + i, (int)0) = cMatrix.m_pTMatrix (nRow, i);
	}

}

void CMatrix::SetMatrixRowNumber(int nRow)
{
	m_nRow = nRow;

	m_pTMatrix.resize (m_nRow, m_nCol);
	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < m_nCol; j++)
		{
			m_pTMatrix(i, j) = (float) 0;
		}
	}

}


void CMatrix::SetMatrixColNumber(int nCol)
{
	m_nCol = nCol;

	m_pTMatrix.resize (m_nRow, m_nCol);
	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < m_nCol; j++)
		{
			m_pTMatrix(i, j) = (float) 0;
		}
	}

}

void CMatrix::Resize(int nRow,int nCol)
{
    this->SetMatrixRowAndCol(nRow, nCol);
}

/////////////////////////////////////////////////////////////////////////
// 设置矩阵的行列数
void CMatrix::SetMatrixRowAndCol(int nRow,int nCol)
{
	m_nRow = nRow;
	m_nCol = nCol;

	// 分配内存
	m_pTMatrix.resize (m_nRow, m_nCol, false);
	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < m_nCol; j++)
		{
			m_pTMatrix(i, j) = (float) 0;
		}
	}

}


/////////////////////////////////////////////////////////////////////////////
// Initialize()
// 矩阵初始化函数,矩阵的行列数目被初始化为零,矩阵中的元素全部初始化为零
/////////////////////////////////////////////////////////////////////////////

void CMatrix::Initialize()
{
	m_nRow = 0;
	m_nCol = 0;

	m_pTMatrix.resize (m_nRow, m_nCol);
	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < m_nCol; j++)
		{
			m_pTMatrix(i, j) = (float) 0;
		}
	}

}

/////////////////////////////////////////////////////////////////////////////
// InitializeZero()
// 矩阵初始化函数,矩阵的行列数目被初始化为零,矩阵中的元素全部初始化为零
/////////////////////////////////////////////////////////////////////////////

void CMatrix::InitializeZero()
{
	m_nRow = 0;
	m_nCol = 0;

	m_pTMatrix.resize (m_nRow, m_nCol);
	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < m_nCol; j++)
		{
			m_pTMatrix(i, j) = (float) 0;
		}
	}

}

/////////////////////////////////////////////////////////////////////////////
// RandomInitialize()
// 将矩阵中的元素随机初始化函数,元素的值在(-1,1)之间的小数
/////////////////////////////////////////////////////////////////////////////

void CMatrix::RandomInitialize(float high, float low)
{
	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < m_nCol; j++)
		{
			//m_pTMatrix (i, j) = ((float) rand() / RAND_MAX) * (high - low) + low;
			m_pTMatrix (i, j) = (float)(rand()%2000) / 1000.0 - 1;
		}
	}

}


/////////////////////////////////////////////////////////////////////////////
// 拷贝矩阵的子矩阵元素到另外一个矩阵中
// Parameter:
//		[out]	cMatrix ----> 矩阵的子矩阵返回的结果
//		[in]	nStartX ----> 子矩阵在矩阵中的起始坐标,对应行,索引从1开始
//		[in]	nStartY ----> 子矩阵在矩阵中的起始坐标,对应列,索引从1开始
/////////////////////////////////////////////////////////////////////////////

void CMatrix::CopySubMatrix(CMatrix& cMatrix,int nStartX,int nStartY)
{
	if((m_nRow  < cMatrix.m_nRow + nStartX ) | (m_nCol  < cMatrix.m_nCol + nStartY))
	{
		throw string("被拷贝的矩阵维数小于要拷贝的矩阵所需要的维数!");
		return;
	}

	for(int i=0;  i < cMatrix.m_nRow; i++)
	{
		for(int j=0; j < cMatrix.m_nCol; j++)
		{
			cMatrix.m_pTMatrix (i, j) = m_pTMatrix (nStartX + i, nStartY + j);
		}
	}

}

/////////////////////////////////////////////////////////////////////////////
// Copy Matrix
//	Parameter:
//		[in]	cMatrix ----> 被拷贝的矩阵
/////////////////////////////////////////////////////////////////////////////

void CMatrix::CopyMatrix(CMatrix cMatrix)
{
	m_nRow	= cMatrix.m_nRow ;
	m_nCol	= cMatrix.m_nCol ;

	m_pTMatrix	= cMatrix.m_pTMatrix ;


}

/////////////////////////////////////////////////////////////////////////////
//	从一个列向量中拷贝数据到一个矩阵中
//	Parameter:
//		[out]	cMatrix ----> 函数的返回结果
//		[in]	nIndex	----> 在列向量中的索引值
//	Notes:
//		被拷贝的对象必须是列向量!!!
/////////////////////////////////////////////////////////////////////////////

void CMatrix::CopySubMatrixFromVector(CMatrix& cMatrix,int nIndex)
{
	if(m_nCol != 1)
	{
		throw string("被拷贝的矩阵不是列向量!!!");
		return;
	}

	for(int j=0; j < cMatrix.m_nCol; j++)
	{
		for(int i=0; i < cMatrix.m_nRow; i++)
		{
			cMatrix.m_pTMatrix (i, j) = m_pTMatrix (nIndex + j * cMatrix.m_nRow + i , (int)0);
		}
	}

}

/////////////////////////////////////////////////////////////////////////////
// 对矩阵进列拓展
//	实现功能:
//		对矩阵的列数进行拓展,nTimes是每列拓展的次数
/////////////////////////////////////////////////////////////////////////////

void CMatrix::nncpyi(const CMatrix &cMatrix, int nTimes)
{
	m_nRow	=	cMatrix.m_nRow ;
	m_nCol	=	cMatrix.m_nCol *	nTimes;

	// 根据空间分配内存
	m_pTMatrix.resize (m_nRow, m_nCol);
	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < m_nCol; j++)
		{
			m_pTMatrix(i, j) = (float) 0;
		}
	}

	// 赋值
	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < cMatrix.m_nCol; j++)
		{
			for(int k=0; k < nTimes; k++)
			{
				m_pTMatrix (i, j * nTimes + k) = cMatrix.m_pTMatrix (i, j);
			}
		}
	}

}

/////////////////////////////////////////////////////////////////////////////
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
/////////////////////////////////////////////////////////////////////////////

void CMatrix::nncpyd(CMatrix &cMatrix)
{
	m_nRow	=	cMatrix.m_nRow ;
	m_nCol	=	cMatrix.m_nCol * cMatrix.m_nRow ;

	// 根据空间分配内存
	m_pTMatrix.resize (m_nRow, m_nCol);
	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < m_nCol; j++)
		{
			m_pTMatrix(i, j) = (float) 0;
		}
	}

	// 给矩阵赋值
	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < cMatrix.m_nCol; j++)
		{
			for(int k=0; k < cMatrix.m_nRow; k++)
			{
				if(i == (j * cMatrix.m_nRow + k) % cMatrix.m_nRow )
					m_pTMatrix (i, j * cMatrix.m_nRow + k) = cMatrix.m_pTMatrix (i, j);
			}
		}
	}

}

/////////////////////////////////////////////////////////////////////////////
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
/////////////////////////////////////////////////////////////////////////////

void CMatrix::nncpy(const CMatrix& cMatrix,int nTimes)
{
	m_nRow = cMatrix.m_nRow ;
	m_nCol = cMatrix.m_nCol * nTimes;

	// 根据空间分配内存
	m_pTMatrix.resize (m_nRow, m_nCol);
	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < m_nCol; j++)
		{
			m_pTMatrix(i, j) = (float) 0;
		}
	}

	// 对矩阵赋值
	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < nTimes; j++)
		{
			for(int k=0; k < cMatrix.m_nCol; k++)
			{
				m_pTMatrix (i, j * cMatrix.m_nCol + k) = cMatrix.m_pTMatrix (i, k);
			}
		}
	}

}

/////////////////////////////////////////////////////////////////////////////
// 对矩阵中所有的元素进行一次非线性变换:
//		变换后的值y与变换前的值的关系是:
//			y = f(x) = 1 / (1 + exp(-x))	( 0 < f(x) < 1)
//
/////////////////////////////////////////////////////////////////////////////

CMatrix CMatrix::Sigmoid()
{
	CMatrix cMatrix(m_nRow, m_nCol);

	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < m_nCol; j++)
		{
			cMatrix.m_pTMatrix (i, j) = 1 / (1 + exp(-m_pTMatrix (i, j)));
		}

	}

	return cMatrix;
}


/////////////////////////////////////////////////////////////////////////////
// 对矩阵中所有的元素进行一次非线性变换:
//		变换后的值y与变换前的值的关系是:
//			y = tanh(x) = (1 - exp(-x)) / (1 + exp(-x))
//					 =  1 - 2 * exp(-x) / (1 + exp(-x))	( -1 < f(x) < 1)
//
/////////////////////////////////////////////////////////////////////////////

CMatrix CMatrix::tanh ()
{
	CMatrix cMatrix(m_nRow, m_nCol);

	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < m_nCol; j++)
		{
			cMatrix.m_pTMatrix (i, j) = 1 - (2 * exp(-m_pTMatrix (i, j))) / (1 + exp(-m_pTMatrix (i, j)));
		}

	}

	return cMatrix;
}

/////////////////////////////////////////////////////////////////////////////
// 对矩阵中所有的元素进行一次非线性变换:
//		变换后的值y与变换前的值的关系是:
//			y = Tansig(x) = 2 / (1 + exp(-2 * x)) -1
/////////////////////////////////////////////////////////////////////////////

CMatrix CMatrix::Tansig()
{
	CMatrix cMatrix(m_nRow, m_nCol);

	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < m_nCol; j++)
		{
			cMatrix.m_pTMatrix (i, j) = 2 / (1 + exp(- 2 * m_pTMatrix (i, j))) - 1;
		}
	}

	return cMatrix;

}

/////////////////////////////////////////////////////////////////////////////
// 对矩阵中所有的元素进行一次非线性变换:
//		变换后的值y与变换前的值的关系是:
//			y = Tansig'(x) = (2 / (1 + exp(-2 * x)) -1)'
//				= (2 / (1 + exp(-2 * x)) -1) * (2 / (1 + exp(-2 * x)) -1) -1
/////////////////////////////////////////////////////////////////////////////

CMatrix CMatrix::TansigDerivative()
{
	CMatrix cMatrix(m_nRow, m_nCol);

	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < m_nCol; j++)
		{
			cMatrix.m_pTMatrix (i, j) = (2 / (1 + exp(- 2 * m_pTMatrix (i, j))) - 1) * (2 / (1 + exp(- 2 * m_pTMatrix (i, j))) - 1) - 1;
		}
	}

	return cMatrix;

}


/////////////////////////////////////////////////////////////////////////////
// 对矩阵中所有的元素进行一次非线性变换:
//		变换后的值y与变换前的值的关系是:
//			y = f'(x) = (1 / (1 + exp(-x)))'	( 0 < f(x) < 1)
//			  = exp(-x)/((1 + exp(-x))*(1 + exp(-x)))
/////////////////////////////////////////////////////////////////////////////

CMatrix CMatrix::SigmoidDerivative()
{
	CMatrix cMatrix(m_nRow, m_nCol);

	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < m_nCol; j++)
		{
			cMatrix.m_pTMatrix (i, j) = exp(-m_pTMatrix (i, j)) / ((1 + exp(-m_pTMatrix (i, j))) * (1 + exp(-m_pTMatrix (i, j))));
		}

	}

	return cMatrix;
}


/////////////////////////////////////////////////////////////////////////////
// 对矩阵中所有的元素进行一次非线性变换:
//		变换后的值y与变换前的值的关系是:
//			y = tanh'(x) = ((1 - exp(-x)) / (1 + exp(-x)))'	( -1 < f(x) < 1)
//					 = 	2*exp(-x)/((1 + exp(-x))*(1 + exp(-x)))
/////////////////////////////////////////////////////////////////////////////

CMatrix CMatrix::tanhDerivative()
{
	CMatrix cMatrix(m_nRow, m_nCol);

	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < m_nCol; j++)
		{
			cMatrix.m_pTMatrix (i, j) = 2 * exp(-m_pTMatrix (i, j)) / ((1 + exp(-m_pTMatrix (i, j))) * (1 + exp(-m_pTMatrix (i, j))));
		}

	}

	return cMatrix;
}


/////////////////////////////////////////////////////////////////////////////
// 实现对点乘操作符的重载
/////////////////////////////////////////////////////////////////////////////

CMatrix CMatrix::operator / (const CMatrix& cMatrixB)
{


	if( (m_nRow != cMatrixB.m_nRow) || (m_nCol != cMatrixB.m_nCol) )
	{
		throw string("两个矩阵的维数不相等,不满足矩阵点乘的条件!");
		return *this;	// return a invalid value
	}

	CMatrix cMatrix(m_nRow, m_nCol);

	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < m_nCol; j++)
		{
			cMatrix.m_pTMatrix (i, j) = m_pTMatrix (i, j) * cMatrixB.m_pTMatrix (i, j);
		}

	}

	return cMatrix;

}

void CMatrix::CopyTo(CMatrix& matrix, int startRow, int startCol)
{
    if(startCol + m_nCol > matrix.GetColCount() || startRow + m_nRow > matrix.GetRowCount())
    {
        throw string("目标矩阵不能容纳源矩阵");

    }

    for(int i = 0; i<m_nRow;i++)
    {
        for(int j=0;j<m_nCol;j++)
        {
            matrix.m_pTMatrix(startRow+i, startCol + j) = m_pTMatrix(i, j);
        }
    }
}

void CMatrix::Print()
{
	for(int i=0;i<m_nRow;i++)
	{
		for(int j=0;j<m_nCol;j++)
		{
			cout.precision(6);
			
			cout << fixed << m_pTMatrix(i, j) << "\t";
		}
		cout << endl;
	}
}

//***************************************************************************
// ordinary function
//

/////////////////////////////////////////////////////////////////////////////
// 重载 'double - CMatrix' 运算符
/////////////////////////////////////////////////////////////////////////////

CMatrix operator - (double nValue, const CMatrix& cMatrixB)
{
	CMatrix	cMatrix(cMatrixB.GetRowCount(), cMatrixB.GetColCount()) ;

	for(int i=0; i < cMatrix.GetRowCount (); i++)
	{
		for(int j=0; j < cMatrix.GetColCount (); j++)
		{
			cMatrix.m_pTMatrix (i, j) = nValue - cMatrixB.m_pTMatrix(i, j);
		}
	}

	return cMatrix;
}



/////////////////////////////////////////////////////////////////////////////
// 矩阵合并运算符
//	合并规则:
//		1. 参与合并运算的两个矩阵的行数必须相等;
//		2. 参与合并的两个矩阵的列数可以不相等;
//		3. 合并后返回的矩阵的行数与参与合并的矩阵的行数相等,列数是参与合并的
//			两个矩阵的列数之和;
/////////////////////////////////////////////////////////////////////////////

CMatrix MergeMatrix(CMatrix& cMatrixA,CMatrix& cMatrixB)
{
	//	条件检测
	if( cMatrixA.GetRowCount () != cMatrixB.GetRowCount () )
	{
		throw string("参与合并的两个矩阵的行数不相等!");

		return cMatrixA;	// return invalid value
	}

	CMatrix cMatrix(cMatrixA.GetRowCount (),cMatrixA.GetColCount () + cMatrixB.GetColCount ());

	for(int i=0; i < cMatrixA.GetRowCount (); i++)
	{
		for(int j=0; j < cMatrixA.GetColCount (); j++)
		{
			cMatrix.m_pTMatrix (i, j) = cMatrixA.m_pTMatrix(i, j);
		}

		for(int k=0; k < cMatrixB.GetColCount (); k++)
		{
			cMatrix.m_pTMatrix (i, cMatrixA.GetColCount () + k) = cMatrixB.m_pTMatrix(i, k);
		}

	}


	return cMatrix;
}




// End of ordinary function
//***************************************************************************
