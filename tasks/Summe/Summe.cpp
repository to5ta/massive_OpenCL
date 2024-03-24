#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include "time.h"

#define SUCCESS 0
#define FAILURE 1

using namespace std;

#include "OpenCLMgr.h"


cl_int summeRek(cl_mem inputBuffer, int size ,OpenCLMgr& mgr)	// size MUST be a multiple of 512 when this function is called
{
    cout << "\nsummeRek(), size: " << size << endl;

	cl_int status;

	// every work group calculates the sum of 512 elements and stores this value into the output buffer
	// the size of the output buffer therefor must contain one element for each workgroup
	// since "size" (the size of the input buffer) is a multiple of 512, this value is size/512
	size_t outsize = size/512;		

	// we want to use the result of the kernel call as input for a rekursiv call of this funktion
	// therefor the  buffer for the result must be a multiple of 512
	// for example: when the size of the input array is 512*1000, 1000 work groups are launched which need to store their results in the output  buffer
	//              the output buffer is created with a size of 1024 such that its size is a multiple of 512 and the buffer can be used as input for a recursiv call
	size_t cloutsize = (outsize+511)/512*512;  // size of buffer for result: outsize enlarged to the next multiple of 512
	cl_int result;

	// create OpenCl buffer for output
	cl_mem outputBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_WRITE , cloutsize * sizeof(cl_int), NULL, NULL);

	// Set kernel arguments.
	status = clSetKernelArg(mgr.summe_kernel, 0, sizeof(cl_mem), (void *)&inputBuffer);
	CHECK_SUCCESS("Error: setting kernel argument 1!")
	status = clSetKernelArg(mgr.summe_kernel, 1, sizeof(cl_mem), (void *)&outputBuffer);
	CHECK_SUCCESS("Error: setting kernel argument 2!")
	
	// Run the kernel.
	size_t global_work_size[1] = {size/2};
	size_t local_work_size[1] = {256};

    cout << "  global_work_size: " << global_work_size[0] << endl;
    cout << "  local_work_size : " << local_work_size[0] << endl;

    status = clEnqueueNDRangeKernel(mgr.commandQueue, mgr.summe_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	CHECK_SUCCESS("Error: enqueuing kernel!")

	if (outsize==1)	// entire sum has been calculated and stored at the beginning of the output buffer
	{
		// Read the output back to host memory.
		status = clEnqueueReadBuffer(mgr.commandQueue, outputBuffer, CL_TRUE, 0, sizeof(cl_int), &result, 0, NULL, NULL);
		CHECK_SUCCESS("Error: sreading buffer!")
	}
	else	// more than one value is stored in the output buffer ==> call function recursively
	{
		if (outsize<cloutsize)	// output buffer is not filled completely and contains "empty" space at the end
		{
			// this empty space must be filled with 0
			// for example: when 1000 output values have been calculated, the 24 additional elements at the end must be set to 0
			cl_int tmp[512] = {0};
			status = clEnqueueWriteBuffer(mgr.commandQueue, outputBuffer, CL_TRUE, outsize * sizeof(cl_int), (cloutsize-outsize)*sizeof(cl_int), &tmp, 0, NULL, NULL);
			CHECK_SUCCESS("Error1: writing buffer!")
		}
		result = summeRek(outputBuffer, cloutsize ,mgr);
	}

	// release buffers
	status = clReleaseMemObject(outputBuffer);

	return result;
}

cl_int summe(cl_int *input, int size, OpenCLMgr& mgr)
{
	cl_int status;
	int result;

	int clsize = (size+511)/512*512;  // next multiple of 512

	cl_mem inputBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_ONLY, clsize * sizeof(cl_int),NULL, NULL);
	status = clEnqueueWriteBuffer(mgr.commandQueue, inputBuffer, CL_TRUE, 0, size * sizeof(cl_int), input, 0, NULL, NULL);
	CHECK_SUCCESS("Error2: writing buffer!")
	if (size<clsize) 
	{
		cl_int tmp[512] = {0};
		status = clEnqueueWriteBuffer(mgr.commandQueue, inputBuffer, CL_TRUE, size * sizeof(cl_int), (clsize-size)*sizeof(cl_int), &tmp, 0, NULL, NULL);
		CHECK_SUCCESS("Error: writing buffer!")
	}

	result = summeRek(inputBuffer, clsize, mgr);

	// release buffers
	status = clReleaseMemObject(inputBuffer);		

	return result;
}



int summe_cpu( cl_int *input, int size ) {
    int summe = 0;

    for(int i=0; i<size; i++){
        summe += input[i];
    }

    return summe;
}




int main(int argc, char* argv[])
{
	OpenCLMgr mgr;

    clock_t start, end;
    double gpu_time_used;
    double cpu_time_used;


    // Initial input,output for the host and create memory objects for the kernel
    int size = 12000000;
    cl_int *input = new cl_int[size];
    for (int i=0 ; i<size ; i++) {
//        input[i] = abs(rand());
        input[i] = 1;
    }

    start = clock();
    // call function
    int result = summe(input, size, mgr);
    end = clock();

    gpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC*1000.f;
    cout << "\n(GPU) Summe:" << result <<"  Duration: " << gpu_time_used <<  " ms" << endl;


    start = clock();
    // call function
    result = summe_cpu(input, size);
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC*1000.f;
    cout << "(CPU) Summe:" << result <<"  Duration: " << cpu_time_used <<  " ms" << endl;




    delete [] input;

	std::cout<<"Passed!\n";
	return SUCCESS;
}