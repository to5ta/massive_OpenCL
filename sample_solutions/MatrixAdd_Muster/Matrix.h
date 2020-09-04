#ifndef MATRIX_H
#define MATRIX_H

#include <CL/cl.h>

#include "OpenCLMgr.h"

class Matrix {
public:
	~Matrix();

	Matrix();
	Matrix(int w, int h);
	Matrix(const Matrix& m);

	Matrix& operator=(const Matrix& m);
	Matrix operator+(const Matrix& m);

	cl_float& operator[](int index);
	cl_float& Elem(int ix, int iy);
	
private:

	cl_int width;
	cl_int height;
	cl_float *data;

	static cl_float dummy;
	static int useGPU;

public:
	static OpenCLMgr * OpenCLmgr;
};


#endif