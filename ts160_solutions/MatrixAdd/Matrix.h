#ifndef MATRIX_H
#define MATRIX_H

#if not defined(_WIN32) and not defined(_WIN64)
#define TESTOK "\x1b[32mOK\x1b[0m"
#define TESTOKRGB "\x1b[38;2;255;0;255mOK\x1b[0m"
#define TESTFAILED "\x1b[31mFAILED\x1b[0m"
#else
#define TESTOK ""
#define TESTOKRGB ""
#define TESTFAILED ""
#endif

#define TOLERANCE 0.001f
#define WARN_TOLERANCE 0

#include <CL/cl.h>
#include <iomanip>

#define MATRIX_NEW_RANDOM       0
#define MATRIX_NEW_ZEROS        1
#define MATRIX_NEW_ONES         2
#define MATRIX_NEW_IDENTITY     3
#include "OpenCLMgr.h"
#include "time.h"




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

    void compare(const Matrix& m){
        if((*this)==m) {
            cout<<this->name<<"=="<<m.name <<"?  ["<< TESTOK <<"]"<<endl;
        }
        else {
            cout << "M_CPU == M_GPU? [" << TESTFAILED << "]\n" << endl;
        }
    }

    void setName(string name){
        this->name = name;
    }

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

    string name;

	static cl_float dummy;
	static int useGPU;
    static int useSharedMemory;

    clock_t time_start, time_end;

    inline void timingBegin() {
        time_start = clock();
    }

    inline void timingEnd() {
        time_end = clock();
        double time_used = ((double) (time_end - time_start)) / CLOCKS_PER_SEC*1000.f;
        cout << "Duration: " << time_used << " ms" << endl;
    }


public:
	static OpenCLMgr * OpenCLmgr;
};


#endif