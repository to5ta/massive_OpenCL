#pragma once

#include <CL/cl.h>
#include <string>
#include <iostream>
#include <string>
#include <map>
#include "OpenCLMgr.h"

#define SUCCESS 0
#define FAILURE 1

#define CHECK_SUCCESS(msg) \
		if (status!=SUCCESS) { \
			cout << msg << endl; \
			return FAILURE; \
		}


class OpenCLMgr
{
public:
	OpenCLMgr();
	~OpenCLMgr();

    cl_int createContext();
	cl_int buildProgram(const char* filepath);
    cl_int createKernels(const char* kernelnames[], cl_uint count);

	int isValid() {return valid;}

	
	cl_context 			context;
	cl_command_queue 	commandQueue;
	cl_program 			program;

	std::map<const char*, cl_kernel> kernels;

	cl_uint numDevices = 0;
	cl_uint deviceNo = 0; // ?1;
	cl_device_id *devices;
	
//	cl_kernel           matadd_kernel;
//    cl_kernel           matmul_kernel;
//    cl_kernel           matmulshared_kernel;

private:
	static int convertToString(const char *filename, std::string& s);

	int init();
	int valid;
};