#include <OpenCL/cl.h>


class OpenCLMgr
{
public:
	OpenCLMgr();
	~OpenCLMgr();

	int isValid() {return valid;}

	
	cl_context context;
	cl_command_queue commandQueue;
	cl_program program;
	
	cl_kernel calcPrefix256kernel;

private:
	static int convertToString(const char *filename, std::string& s);

	int init();
	int valid;
};
