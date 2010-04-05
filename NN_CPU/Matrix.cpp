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

	// �Զ��������ֵ
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
	// Ҫ���������ӵ�����: ������Ŀ���!
	if(m_nRow != cMatrixB.m_nRow || m_nCol != cMatrixB.m_nCol )
	{
		throw string("ִ����ӵ���������ά�������!");
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
	// Ҫ���������ӵ�����: ��������Ŀ���!
	if(m_nRow != cMatrixB.m_nRow || m_nCol != cMatrixB.m_nCol )
	{
		throw string("ִ���������������ά�������!");
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
		throw string("ִ����˵���������ά����������˵�����!");
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
		throw string("�Ⱥ��������ߵľ����ά�������!");
		return *this;	// return invalid value
	}*/
    m_pTMatrix.resize (cMatrixB.m_nRow, cMatrixB.m_nCol);
	// ��������ֵ
	m_nRow = cMatrixB.m_nRow ;
	m_nCol = cMatrixB.m_nCol ;
	m_pTMatrix = cMatrixB.m_pTMatrix ;


	return *this;
}

CMatrix& CMatrix::operator += (CMatrix& cMatrixB)
{
	if(m_nRow != cMatrixB.m_nRow || m_nCol != cMatrixB.m_nCol )
	{
		//printf("����!ִ����ӵ���������ά�������!\n");
		throw string("����������߾����ά�������!");
		return *this;	// return invalid value
	}

	// ��ֵ����
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
/////////////////////////////////////////////////////////////////////////////

CMatrix CMatrix::MergeColumnsToColumnVector()
{
	CMatrix cMatrix(m_nRow * m_nCol,(int)1);

	// �Ծ���ֵ
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
	// �ο���Ŀ: �������ֵ���� --->ʩ���� �¹�֦
	/////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////
	// �ж��Ƿ��ǿ�����:
	//		������һ���Ƿ���!!!

	if ( m_nRow != m_nCol)
	{
		//printf("����!����������������,�Ƿǿ�����!\n");
		throw string("����������������,�Ƿǿ�����!");
	}

	CMatrix cMatrix(m_nRow, m_nCol);

	//***********************************************************************
	// ˼·:(�ǳ���˼ά!)
	//		��̬������������(2*m_nCol)���洢ÿ�ν������������ֵ
	//		������û���н�������¼��������,
	//		1.û�����н����������������,��SwapMatrixRow()������
	//		��⵽����ֵ�����������,��SwapMatrixCol()������Ҳһ��,
	//		��⵽����ֵ�����������,��ռ��ϵͳ��Դ;
	//		2.����ȵľͽ���
	//***********************************************************************

	//	�����ڴ�
	int *pIntArray = new int [2*m_nCol];

	// nSetp --- Լ������,����չ��
	for(int k=0; k < cMatrix.m_nCol; k++)
	{
		/////////////////////////////////////////////////////////////////////
		// �����н��� ---> ��Ϸ����:
		// Ϊ��֤������̵���ֵ�ȶ���,�ڵ�k��Լ��ʱ,����{a(ik)}|i=k->n��ѡ��
		// ģ�������ΪԼ����Ԫ��,������������Ӧ����

		// �����Ԫ��
		double nMaxElement = cMatrix.m_pTMatrix(k, k);
		// �����Ԫ�����ڵ�����
		int nMainRow = k;

		for(int nCount = k+1; nCount < cMatrix.m_nCol; nCount++)
		{
			if( fabs(nMaxElement) < fabs(cMatrix.m_pTMatrix(nCount, k)) )
			{
				nMaxElement = cMatrix.m_pTMatrix (nCount, k);
				nMainRow = nCount;
			}
		}

		// ������������������������
		pIntArray [2*k] = k;
		pIntArray [2*k+1] = nMainRow;


		// ������
		cMatrix.SwapMatrixRow(k,nMainRow);

		//Display();

		//	�ж��Ƿ��ǿ�����
		if(cMatrix.m_pTMatrix (k, k) == 0)
		{
			//printf("����!�˾���Ϊ�ǿ�����!\n");
			throw string("�˾���Ϊ�ǿ�����,û�������!");
		}

		cMatrix.m_pTMatrix (k, k) = 1/(cMatrix.m_pTMatrix(k, k));


		// ������
		for(int i=0; i < cMatrix.m_nRow; i++)
		{
			if( i != k)
				cMatrix.m_pTMatrix (i, k) = -(cMatrix.m_pTMatrix (k, k)) * (cMatrix.m_pTMatrix(i, k));

			//int nTempValue = m_pTMatrix(i, k);

		}

		//printf("\n");

		// Լ��������
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

		// ������
		for(int j=0; j < cMatrix.m_nCol; j++)
		{
			if( j != k)
				cMatrix.m_pTMatrix (k, j) = (cMatrix.m_pTMatrix (k, k)) * (cMatrix.m_pTMatrix (k, j));

		}

	}


	/////////////////////////////////////////////////////////////////////////
	// �����н��� ---> �Խ����к�ľ�������н��� ---> ��ԭ����
	// ��Ϸ����:
	// ����ʼ�����н��е��н��� ---> �������Ӧ���н������л�ԭ,���ɵõ������
	// �����

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
		throw string("�޷���������");
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
// �Ծ����е�Ԫ�ؽ���һ�β���:
//		ʹ�����Ϊ��λ��
/////////////////////////////////////////////////////////////////////////////

void CMatrix::Eye()
{
	// Verify whether the rows is equal to the columns or not
	if(m_nRow != m_nCol)
	{
		throw string("�˾���������������!����ת��Ϊ��λ��!");
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
//		CMatrix& cMatrix:		������������Դ
//		int nIndex:	�������������ڶ����еĿ�ʼ����λ��,��0��ʼ
// Purpose:
//		This function will copy all the data of the cMatrix
// Notes:
//		�˶��������������!!!
/////////////////////////////////////////////////////////////////////////////

void CMatrix::GetMatrixData(CMatrix& cMatrix, int nIndex)
{
	if(m_nCol != 1)
	{
		throw string("�����ľ�����������!");
		return;
	}

	if((m_nRow - nIndex) < (cMatrix.m_nRow * cMatrix.m_nCol))
	{
		throw string("��������Ŀռ���������!");
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
//		CMatrix& cMatrix:		�����ľ���
//		int nIndex:	�������������ڶ����еĿ�ʼ����λ��
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
		throw string("�����������������,����������!");
		return;
	}

	// Verify whether the number of the object element is enough to be copyed
	if((m_nRow - nIndex) < (cMatrix.m_nRow * cMatrix.m_nCol))
	{
		throw string("�����е�Ԫ����������!");
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
//		CMatrix& cMatrix:		�����ľ���
//		int nIndex:	�������������ڶ����еĿ�ʼ����λ��
//		int nRow:		�����������ڱ��������е�������
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
		throw string("�����������������,����������!");
		return;
	}

	// Verify whether the number of the object element is enough to be copyed
	if((m_nRow - nIndex) < cMatrix.m_nCol )
	{
		throw string("�����Ԫ����������!");
		return;
	}

	for(int i=0; i < cMatrix.m_nCol; i++)
	{
		cMatrix.m_pTMatrix (nRow, i) = m_pTMatrix (nIndex + i, (int)0);
	}

}


/////////////////////////////////////////////////////////////////////////////
// Parameter:
//		CMatrix& cMatrix:		������������Դ
//		int nIndex:	�������������ڶ����еĿ�ʼ����λ��
//		int nRow:		�������������ڱ����������е�������(��0��ʼ)
// Purpose:
//		This function will copy all the data of the cMatrix
//	Notes:
//		�˶��������������!!!
/////////////////////////////////////////////////////////////////////////////

void CMatrix::GetMatrixRowData(CMatrix& cMatrix, int nIndex, int nRow)
{
	if(m_nCol != 1)
	{
		throw string("�����ľ�����������!");
		return;
	}

	if((m_nRow - nIndex) < cMatrix.m_nCol)
	{
		throw string("��������Ŀռ���������!");
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
// ���þ����������
void CMatrix::SetMatrixRowAndCol(int nRow,int nCol)
{
	m_nRow = nRow;
	m_nCol = nCol;

	// �����ڴ�
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
// �����ʼ������,�����������Ŀ����ʼ��Ϊ��,�����е�Ԫ��ȫ����ʼ��Ϊ��
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
// �����ʼ������,�����������Ŀ����ʼ��Ϊ��,�����е�Ԫ��ȫ����ʼ��Ϊ��
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
// �������е�Ԫ�������ʼ������,Ԫ�ص�ֵ��(-1,1)֮���С��
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
// ����������Ӿ���Ԫ�ص�����һ��������
// Parameter:
//		[out]	cMatrix ----> ������Ӿ��󷵻صĽ��
//		[in]	nStartX ----> �Ӿ����ھ����е���ʼ����,��Ӧ��,������1��ʼ
//		[in]	nStartY ----> �Ӿ����ھ����е���ʼ����,��Ӧ��,������1��ʼ
/////////////////////////////////////////////////////////////////////////////

void CMatrix::CopySubMatrix(CMatrix& cMatrix,int nStartX,int nStartY)
{
	if((m_nRow  < cMatrix.m_nRow + nStartX ) | (m_nCol  < cMatrix.m_nCol + nStartY))
	{
		throw string("�������ľ���ά��С��Ҫ�����ľ�������Ҫ��ά��!");
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
//		[in]	cMatrix ----> �������ľ���
/////////////////////////////////////////////////////////////////////////////

void CMatrix::CopyMatrix(CMatrix cMatrix)
{
	m_nRow	= cMatrix.m_nRow ;
	m_nCol	= cMatrix.m_nCol ;

	m_pTMatrix	= cMatrix.m_pTMatrix ;


}

/////////////////////////////////////////////////////////////////////////////
//	��һ���������п������ݵ�һ��������
//	Parameter:
//		[out]	cMatrix ----> �����ķ��ؽ��
//		[in]	nIndex	----> ���������е�����ֵ
//	Notes:
//		�������Ķ��������������!!!
/////////////////////////////////////////////////////////////////////////////

void CMatrix::CopySubMatrixFromVector(CMatrix& cMatrix,int nIndex)
{
	if(m_nCol != 1)
	{
		throw string("�������ľ�����������!!!");
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
// �Ծ��������չ
//	ʵ�ֹ���:
//		�Ծ��������������չ,nTimes��ÿ����չ�Ĵ���
/////////////////////////////////////////////////////////////////////////////

void CMatrix::nncpyi(const CMatrix &cMatrix, int nTimes)
{
	m_nRow	=	cMatrix.m_nRow ;
	m_nCol	=	cMatrix.m_nCol *	nTimes;

	// ���ݿռ�����ڴ�
	m_pTMatrix.resize (m_nRow, m_nCol);
	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < m_nCol; j++)
		{
			m_pTMatrix(i, j) = (float) 0;
		}
	}

	// ��ֵ
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
/////////////////////////////////////////////////////////////////////////////

void CMatrix::nncpyd(CMatrix &cMatrix)
{
	m_nRow	=	cMatrix.m_nRow ;
	m_nCol	=	cMatrix.m_nCol * cMatrix.m_nRow ;

	// ���ݿռ�����ڴ�
	m_pTMatrix.resize (m_nRow, m_nCol);
	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < m_nCol; j++)
		{
			m_pTMatrix(i, j) = (float) 0;
		}
	}

	// ������ֵ
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
/////////////////////////////////////////////////////////////////////////////

void CMatrix::nncpy(const CMatrix& cMatrix,int nTimes)
{
	m_nRow = cMatrix.m_nRow ;
	m_nCol = cMatrix.m_nCol * nTimes;

	// ���ݿռ�����ڴ�
	m_pTMatrix.resize (m_nRow, m_nCol);
	for(int i=0; i < m_nRow; i++)
	{
		for(int j=0; j < m_nCol; j++)
		{
			m_pTMatrix(i, j) = (float) 0;
		}
	}

	// �Ծ���ֵ
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
// �Ծ��������е�Ԫ�ؽ���һ�η����Ա任:
//		�任���ֵy��任ǰ��ֵ�Ĺ�ϵ��:
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
// �Ծ��������е�Ԫ�ؽ���һ�η����Ա任:
//		�任���ֵy��任ǰ��ֵ�Ĺ�ϵ��:
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
// �Ծ��������е�Ԫ�ؽ���һ�η����Ա任:
//		�任���ֵy��任ǰ��ֵ�Ĺ�ϵ��:
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
// �Ծ��������е�Ԫ�ؽ���һ�η����Ա任:
//		�任���ֵy��任ǰ��ֵ�Ĺ�ϵ��:
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
// �Ծ��������е�Ԫ�ؽ���һ�η����Ա任:
//		�任���ֵy��任ǰ��ֵ�Ĺ�ϵ��:
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
// �Ծ��������е�Ԫ�ؽ���һ�η����Ա任:
//		�任���ֵy��任ǰ��ֵ�Ĺ�ϵ��:
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
// ʵ�ֶԵ�˲�����������
/////////////////////////////////////////////////////////////////////////////

CMatrix CMatrix::operator / (const CMatrix& cMatrixB)
{


	if( (m_nRow != cMatrixB.m_nRow) || (m_nCol != cMatrixB.m_nCol) )
	{
		throw string("���������ά�������,����������˵�����!");
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
        throw string("Ŀ�����������Դ����");

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
// ���� 'double - CMatrix' �����
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
// ����ϲ������
//	�ϲ�����:
//		1. ����ϲ��������������������������;
//		2. ����ϲ�������������������Բ����;
//		3. �ϲ��󷵻صľ�������������ϲ��ľ�����������,�����ǲ���ϲ���
//			�������������֮��;
/////////////////////////////////////////////////////////////////////////////

CMatrix MergeMatrix(CMatrix& cMatrixA,CMatrix& cMatrixB)
{
	//	�������
	if( cMatrixA.GetRowCount () != cMatrixB.GetRowCount () )
	{
		throw string("����ϲ���������������������!");

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
