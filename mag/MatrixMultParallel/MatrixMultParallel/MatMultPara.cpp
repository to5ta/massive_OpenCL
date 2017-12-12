/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

// For clarity,error checking has been omitted.

#include <OpenCL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>

#define SUCCESS 0
#define FAILURE 1

using namespace std;

#include "OpenCLMgr.h"


void makeMat(cl_int mat[], cl_int width, cl_int height)
{
    for(int i = 0; i < height*width; i++)
    {
        mat[i] = rand() % 100;
    }
}


int main(int argc, char* argv[])
{
	OpenCLMgr mgr;


	// Initial input,output for the host and create memory objects for the kernel
    const cl_int K = 1008;
    const cl_int M = 1008;
    const cl_int KM = K * M;
    cl_int matA[KM] = {};
    makeMat(matA, K, M);
    
    const cl_int N = 1008;
    const cl_int NK = N * K;
    cl_int matB[NK] = {};
    makeMat(matB, N, K);
    
    size_t MN = M * N;
    
    int *output = (int*) malloc(MN);

    cl_int status;
    
    cl_mem inMatA = clCreateBuffer(mgr.context, CL_MEM_READ_ONLY, KM * sizeof(cl_int), NULL, NULL);
    status = clEnqueueWriteBuffer(mgr.commandQueue, inMatA, CL_TRUE, 0, KM * sizeof(cl_int), matA, 0, NULL, NULL);
    CHECK_SUCCESS("Error: writing buffer!")

    cl_mem inMatB = clCreateBuffer(mgr.context, CL_MEM_READ_ONLY, NK * sizeof(cl_int), NULL, NULL);
    status = clEnqueueWriteBuffer(mgr.commandQueue, inMatB, CL_TRUE, 0, NK * sizeof(cl_int), matB, 0, NULL, NULL);
    CHECK_SUCCESS("Error: writing buffer!")

    cl_mem outMat = clCreateBuffer(mgr.context, CL_MEM_WRITE_ONLY, MN * sizeof(cl_int), NULL, NULL);
    
    status = clSetKernelArg(mgr.matMultPara_kernel, 0, sizeof(cl_mem), (void *)&M);
    status = clSetKernelArg(mgr.matMultPara_kernel, 1, sizeof(cl_mem), (void *)&N);
    status = clSetKernelArg(mgr.matMultPara_kernel, 2, sizeof(cl_mem), (void *)&K);
    status = clSetKernelArg(mgr.matMultPara_kernel, 3, sizeof(cl_mem), (void *)&inMatA);
    status = clSetKernelArg(mgr.matMultPara_kernel, 4, sizeof(cl_mem), (void *)&inMatB);
    status = clSetKernelArg(mgr.matMultPara_kernel, 5, sizeof(cl_mem), (void *)&outMat);
    CHECK_SUCCESS("Error: setting kernel arguments!")
    
    const cl_int TS = 8;
    size_t global_work_size[2] = {M, N};
    size_t local_work_size[2] = {TS, TS};
    status = clEnqueueNDRangeKernel(mgr.commandQueue, mgr.matMultPara_kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    CHECK_SUCCESS("Error: running kernel!")
    
    /*Step 11: Read the cout put back to host memory.*/
    status = clEnqueueReadBuffer(mgr.commandQueue, outMat, CL_TRUE, 0, MN * sizeof(cl_int), output, 0, NULL, NULL);
    


    int idx = 0;
    cout << "Mat Mult:" << endl;
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
            cout << output[idx] << " ";
            idx++;
        }
        cout << endl;
    }
	
    
    // release buffers
//    status = clReleaseMemObject(inMatA);
//    status = clReleaseMemObject(inMatB);
//    status = clReleaseMemObject(outMat);
    
	std::cout << "Passed!\n" << endl;
	return SUCCESS;
}
