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


int main(int argc, char* argv[])
{
	OpenCLMgr mgr;


	// Initial input,output for the host and create memory objects for the kernel
	cl_int matA[6] = {1, 2, 3, 4, 5, 6};
    cl_int sizeAx = 2;
    cl_int sizeAy = 3;
    cl_int matB[6] = {1, 2, 3, 4, 5, 6};
    cl_int sizeBx = 3;
    cl_int sizeBy = 2;
    
    int sizeA = sizeAx * sizeAy;
    int sizeB = sizeBx * sizeBy;
    size_t sizeOut = sizeAy * sizeBx;
    
    int *output = (int*) malloc(sizeOut);

    cl_int status;
    
    cl_mem inMatA = clCreateBuffer(mgr.context, CL_MEM_READ_ONLY, sizeA * sizeof(cl_int), NULL, NULL);
    status = clEnqueueWriteBuffer(mgr.commandQueue, inMatA, CL_TRUE, 0, sizeA * sizeof(cl_int), matA, 0, NULL, NULL);
    CHECK_SUCCESS("Error: writing buffer!")

    cl_mem inMatB = clCreateBuffer(mgr.context, CL_MEM_READ_ONLY, sizeB * sizeof(cl_int), NULL, NULL);
    status = clEnqueueWriteBuffer(mgr.commandQueue, inMatB, CL_TRUE, 0, sizeB * sizeof(cl_int), matB, 0, NULL, NULL);
    CHECK_SUCCESS("Error: writing buffer!")

    cl_mem outMat = clCreateBuffer(mgr.context, CL_MEM_WRITE_ONLY, sizeOut * sizeof(cl_int), NULL, NULL);

    
    status = clSetKernelArg(mgr.matMultPara_kernel, 0, sizeof(cl_mem), (void *)&inMatA);
    CHECK_SUCCESS("Error: setting kernel argument 0!")
    status = clSetKernelArg(mgr.matMultPara_kernel, 1, sizeof(cl_mem), (void *)&sizeAx);
    CHECK_SUCCESS("Error: setting kernel argument 1!")
    status = clSetKernelArg(mgr.matMultPara_kernel, 2, sizeof(cl_mem), (void *)&sizeAy);
    CHECK_SUCCESS("Error: setting kernel argument 2!")
    status = clSetKernelArg(mgr.matMultPara_kernel, 3, sizeof(cl_mem), (void *)&inMatB);
    CHECK_SUCCESS("Error: setting kernel argument 3!")
    status = clSetKernelArg(mgr.matMultPara_kernel, 4, sizeof(cl_mem), (void *)&sizeBx);
    CHECK_SUCCESS("Error: setting kernel argument 4!")
    status = clSetKernelArg(mgr.matMultPara_kernel, 5, sizeof(cl_mem), (void *)&sizeAy);
    CHECK_SUCCESS("Error: setting kernel argument 5!")
    status = clSetKernelArg(mgr.matMultPara_kernel, 6, sizeof(cl_mem), (void *)&outMat);
    CHECK_SUCCESS("Error: setting kernel out argument 6!")
    status = clSetKernelArg(mgr.matMultPara_kernel, 7, sizeof(cl_mem), (void *)&sizeOut);
    CHECK_SUCCESS("Error: setting kernel argument 7!")
    
    
    size_t global_work_size[1] = {sizeOut};
    size_t local_work_size[1] = {sizeOut};
    status = clEnqueueNDRangeKernel(mgr.commandQueue, mgr.matMultPara_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    CHECK_SUCCESS("Error: running kernel!")
    
    /*Step 11: Read the cout put back to host memory.*/
    status = clEnqueueReadBuffer(mgr.commandQueue, outMat, CL_TRUE, 0, sizeOut * sizeof(cl_int), output, 0, NULL, NULL);
    
    // release buffers
    status = clReleaseMemObject(inMatA);
    status = clReleaseMemObject(inMatB);
    status = clReleaseMemObject(outMat);

    int idx = 0;
    cout << "Mat Mult:" << endl;
    for(int i = 0; i < sizeAy; i++)
    {
        for(int j = 0; j < sizeBx; j++)
        {
            cout << output[idx] << " ";
            idx++;
        }
        cout << endl;
    }
	
	std::cout<<"Passed!\n";
	return SUCCESS;
}
