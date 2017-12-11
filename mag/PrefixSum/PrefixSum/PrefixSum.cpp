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

#define CHECK_SUCCESS(msg) \
		if (status!=SUCCESS) { \
			cout << msg << endl; \
			return FAILURE; \
		}


int calcPrefixSums(OpenCLMgr &mgr, int feld[], int PreFix[])	// arrays must have a size of 256
{
	cl_int status;

	// create OpenCL buffers 
	cl_mem inputBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_ONLY, 256 * sizeof(int), NULL, &status);
	CHECK_SUCCESS("Error: creating inputBuffer")
	cl_mem outputBuffer = clCreateBuffer(mgr.context, CL_MEM_WRITE_ONLY , 256 * sizeof(int), NULL, &status);
	CHECK_SUCCESS("Error: creating outputBuffer")

	// copy array to OpenCL buffer
	status = clEnqueueWriteBuffer(mgr.commandQueue, inputBuffer, CL_FALSE, 0, 256 * sizeof(int), feld, 0, NULL, NULL);
	CHECK_SUCCESS("Error: copying data to inputBuffer")

	// Sets Kernel arguments.
	status = clSetKernelArg(mgr.calcPrefix256kernel, 0, sizeof(cl_mem), (void *)&inputBuffer);
	CHECK_SUCCESS("Error: setting kernel argument 0")
	status = clSetKernelArg(mgr.calcPrefix256kernel, 1, sizeof(cl_mem), (void *)&outputBuffer);
	CHECK_SUCCESS("Error: setting kernel argument 1")
	
	// Running the kernel.
	cl_event ev;
	size_t global_work_size[1] = {256};
	size_t local_work_size[1] = {256};
	status = clEnqueueNDRangeKernel(mgr.commandQueue, mgr.calcPrefix256kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &ev);
	CHECK_SUCCESS("Error: enqueuing kernel")

	// Profiling: determine kernel execution time
	cl_ulong start, end;
	clWaitForEvents(1, &ev);
	status = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL); 
	status = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);  
	cl_ulong time_micros = (end-start)/1000;

	// read result back to host
	status = clEnqueueReadBuffer(mgr.commandQueue, outputBuffer, CL_TRUE, 0, 256 * sizeof(int), PreFix, 0, NULL, NULL);
	CHECK_SUCCESS("Error: copying data from output buffer to host")
	
    cout<<PreFix<<endl;

	// release buffers
	status = clReleaseMemObject(inputBuffer);		//Release mem object.
	CHECK_SUCCESS("Error: releasing inputBuffer")
	status = clReleaseMemObject(outputBuffer);
	CHECK_SUCCESS("Error: releasing outputBuffer")

	return SUCCESS;
}
	
int main(int argc, char* argv[])
{
	OpenCLMgr mgr;

	if (!mgr.isValid())
	{
		cout << "Initialization of OpenCL failes" << endl;
		return FAILURE;
	}

	int feld[256];
	int PreFix[256];

	int i;
	for (i=0 ; i<256 ; i++)
		feld[i]=1;

	cl_int status = calcPrefixSums(mgr, feld, PreFix);

	if (status==SUCCESS)
		std::cout<<"Passed!\n";

	return SUCCESS;
}
