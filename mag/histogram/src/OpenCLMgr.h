#pragma once


#include <OpenCL/cl.h>
#include <string>
#include <iostream>
#include <string>

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

	int isValid() {return valid;}

	
	cl_context 			context;
	cl_command_queue 	commandQueue;
	cl_program 			program;
	
	cl_kernel           calc_kernel;
    cl_kernel           reduce_kernel;


private:
	static int convertToString(const char *filename, std::string& s);

	int init();
	int valid;
};