/**********************************************************************
Copyright �2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

�	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
�	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_BRIGHTGREEN   "\x1b[1;32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"


// For clarity,error checking has been omitted.

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
    cout << "Starting Program.." << endl;
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

       int size = 1700;
//    int size = 1000;
//    int size = 1440;

    for(int i=0; i<5; i++){

        size = size*2;

//         float dat[size*size] = {0};

        Matrix F(size,size, MATRIX_NEW_RANDOM);
        Matrix G(size,size, MATRIX_NEW_RANDOM);

//        Matrix F(size,size, MATRIX_NEW_ONES);
//        Matrix G(size,size, MATRIX_NEW_IDENTITY);
//

        Matrix M_GPU, M_GPU_SHARED, M_CPU;
        M_GPU.setName("GPU");
        M_CPU.setName("CPU");
        M_GPU_SHARED.setName("GPU_SHARED");

        cout << "------["<< size <<"x"<< size <<"]*["<< size <<"x"<< size <<"]--------------------" << endl;

        Matrix::setUseGPU(1);
        Matrix::setUseSharedMemory(0);
        M_GPU = F*G;

        Matrix::setUseGPU(1);
        Matrix::setUseSharedMemory(1);
        M_GPU_SHARED = F*G;

//        Matrix::setUseGPU(0);
//        M_CPU = F*G;

        cout << endl;

        if(size<100){
            F.plot("F");
            G.plot("G");

            M_CPU.plot("CPU");
//            M_GPU.plot("GPU");
            M_GPU_SHARED.plot("GPU_SHARED");
        }


        M_GPU.compare(M_GPU_SHARED);
//        M_CPU.compare(M_GPU);
//        M_CPU.compare(M_GPU_SHARED);

    }
//        M_GPU.plot("GPU");
//        M_GPU_SHARED.plot("GPU SHARED");


    std::cout<<"Passed!\n";
	return SUCCESS;
}