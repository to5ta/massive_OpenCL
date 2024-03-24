#pragma once

#include <CL/cl.h>
#include <string.h>
#include <iostream>
#include <string.h>
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
	OpenCLMgr(cl_command_queue_properties queue_flags);
	~OpenCLMgr();

    cl_int createContext(cl_command_queue_properties queue_flags);

    cl_int loadFile(const char* filepath);
    cl_int buildProgram();

    cl_int createKernels(const char* kernelnames[], cl_uint count);

    cl_int setVariable(const char* DEF_NAME,
                       const int   value);


	int isValid() {return valid;}

	cl_context 			context;
	cl_command_queue 	commandQueue;
	cl_program 			program;

    std::string sourceStr;
    char *source;

	std::map<const char*, cl_kernel> kernels;

	cl_uint numDevices = 0;
	cl_uint deviceNo = 0; // ?1;
	cl_device_id *devices;

	cl_ulong maxMem=0;
	
//	cl_kernel           matadd_kernel;
//    cl_kernel           matmul_kernel;
//    cl_kernel           matmulshared_kernel;

private:
	static int convertToString(const char *filename, std::string& s);

	int init(cl_command_queue_properties queue_flags);
	int valid;
};