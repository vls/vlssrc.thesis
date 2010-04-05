#ifndef TOOLS_H_INCLUDED
#define TOOLS_H_INCLUDED

template <class T>
class MyArray
{
public:
	MyArray(int length)
	{
		this->arr = new T[length];
		this->length = length;
	}
	~MyArray()
	{
		delete[] this->arr;
	}
	T* GetPtr()
	{
		return this->arr;
	}
	int GetLength()
	{
		return this->length;
	}
	T& operator[] (int index)
	{
		return this->arr[index];
	}
private:
	T* arr;
	int length;

};

#endif