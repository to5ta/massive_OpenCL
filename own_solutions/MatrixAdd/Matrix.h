#ifndef MATRIX_H
#define MATRIX_H

#include <CL/cl.h>
#define MATRIX_NEW_RANDOM       0
#define MATRIX_NEW_ZEROS        1
#define MATRIX_NEW_ONES         2
#define MATRIX_NEW_IDENTITY     3
#include "OpenCLMgr.h"



class Matrix {
public:
	~Matrix();

	Matrix();
	Matrix(int w, int h);
	Matrix(const Matrix& m);
    Matrix(int w, int h, float *data);
	Matrix(int w, int h, int TYPE);

	Matrix& operator=(const Matrix& m);
	Matrix operator+(const Matrix& m);
    Matrix operator*(const Matrix& m);
    int operator==(const Matrix& m);

	cl_float& operator[](int index);
	cl_float& Elem(int ix, int iy);


    void plot(void);

    void plot(string s){
        cout << s << endl;
        plot();
    }

	void info(void){
		printf("Width:  %5i\n", width);
		printf("Height: %5i\n", height);
	}

	void info(string s){
		cout << s << endl;
		info();
	}

    static void setUseGPU(int b){
        useGPU = b;
        // printf("set useGPU to %i", b);
        return;
    }

    static void setUseSharedMemory(int b){
        useSharedMemory = b;
        return;
    }
private:

	cl_int width;
	cl_int height;
	cl_float *data;

	static cl_float dummy;
	static int useGPU;
    static int useSharedMemory;


public:
	static OpenCLMgr * OpenCLmgr;
};


#endif