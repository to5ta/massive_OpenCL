
#include <stdlib.h>
#include <string.h>

using namespace std;

#include "Matrix.h"
#include "OpenCLMgr.h"

#define INDEX(M,x,y) ((y)*((M).width)+(x))

cl_float Matrix::dummy;
int Matrix::useGPU = 1;
OpenCLMgr* Matrix::OpenCLmgr = NULL;;


Matrix::Matrix()
{ 
	width = height = 0;
	data = NULL;
}

Matrix::Matrix(int w, int h)
{
	width = w;
	height = h;
	data = new cl_float[width*height];
}

Matrix::Matrix(const Matrix& m)
{
	width = m.width;
	height = m.height;
	data = new cl_float[width*height];
	memcpy(data, m.data, width*height*sizeof(cl_float));
}

Matrix::~Matrix()
{
	delete [] data;
}

Matrix& Matrix::operator=(const Matrix& m)
{
	if (this!=&m)
	{
		delete [] data;
		width = m.width;
		height = m.height;
		data = new cl_float[width*height];
		memcpy(data, m.data, width*height*sizeof(cl_float));
	}
	return *this;
}

Matrix Matrix::operator+(const Matrix& m)
{
	if (m.width==width && m.height==height)
	{
		Matrix result(width, height);

		if (!useGPU)
		{
			// use CPU
			for (int i=0 ; i<width*height ; i++)
				result.data[i]=data[i]+m.data[i];
			return result;
		}
		else	// use GPU
		{
			// TODO

			return result;
		}
	}
	else
	{
		return Matrix();
	}
}

cl_float& Matrix::operator[](int index)
{
	if (index>=0 && index<width*height)
	{
		return data[index];
	}
	else
	{
		return dummy;
	}
}
	

cl_float& Matrix::Elem(int ix, int iy)
{
	int index = INDEX(*this, ix, iy);
	return (*this)[index];
}
