
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_BRIGHTGREEN   "\x1b[1;32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"


#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>


#define SUCCESS 0
#define FAILURE 1

using namespace std;

#include "OpenCLMgr.h"
#include "Matrix.h"

int main(int argc, char* argv[])
{
	OpenCLMgr mgr;

	Matrix::OpenCLmgr = &mgr;

	Matrix A(5,5);
	Matrix B(5,5);
	Matrix C;

    Matrix::setUseGPU(1);


    float data[] = {0.496977467838,  0.913963293057,  0.652302605086,  0.179985943407,  0.899937338564, \
                   0.0800485928486, 0.0575882535388,  0.423745480297,  0.125236501884,  0.847683123528, \
                    0.757251050449,  0.940613928412,  0.898778373763,  0.489139256615,  0.183107854405,  \
                     0.60116239376,  0.123433164057,  0.22810878442,   0.346700684011,  0.155464970501,  \
                     0.520739164003, 0.450725701385,  0.216825612261,  0.79675506135,   0.317620970921 };

    Matrix R(5,5, data);

	int x,y;
	for (x=0 ; x<5 ; x++)
		for (y=0 ; y<5 ; y++)
			A.Elem(x,y)=1.0f;


    for (x=0 ; x<5 ; x++)
    {
        for (y=0 ; y<5 ; y++)
            B.Elem(x,y)=0.0f;
        B.Elem(x,x)=2.0f;
    }


//    A.info("A");
//    B.info("B");
//    C.info("C");
//    C = R*B;

//    R.plot();
//    B.plot();
//    C.plot();

    int size = 512;

    for(int i=0; i<1; i++){

//        size = size*2;

        float dat[size*size] = {0};
        Matrix F(size,size, MATRIX_NEW_RANDOM);
        Matrix G(size,size, MATRIX_NEW_RANDOM);

        Matrix M_GPU, M_GPU_SHARED, M_CPU;
        M_GPU.setName("GPU");
        M_CPU.setName("CPU");
        M_GPU_SHARED.setName("GPU_SHARED");

        cout << "["<< size <<"x"<< size <<"]*["<< size <<"x"<< size <<"]--------------------" << endl;

        Matrix::setUseGPU(1);
        Matrix::setUseSharedMemory(0);
        M_GPU = F*G;

        Matrix::setUseGPU(1);
        Matrix::setUseSharedMemory(1);
        M_GPU_SHARED = F*G;

        Matrix::setUseGPU(0);
        M_CPU = F*G;

        M_CPU.compare(M_GPU);
        M_CPU.compare(M_GPU_SHARED);

        if(size<20){
            F.plot("F");
            G.plot("G");

            M_CPU.plot("CPU");
            M_GPU.plot("GPU");
            M_GPU_SHARED.plot("GPU_SHARED");
        }


    }
//        M_GPU.plot("GPU");
//        M_GPU_SHARED.plot("GPU SHARED");







    std::cout<<"Passed!\n";
	return SUCCESS;
}