
#include <stdlib.h>
#include <string.h>

using namespace std;

#include "Matrix.h"
#include "OpenCLMgr.h"

#define INDEX(M,x,y) ((y)*((M).width)+(x))

cl_float Matrix::dummy;
int Matrix::useGPU = 1;
OpenCLMgr* Matrix::OpenCLmgr = NULL;;


Matrix::Matrix()
{ 
	width = height = 0;
	data = NULL;
}

Matrix::Matrix(int w, int h)
{
	width = w;
	height = h;
	data = new cl_float[width*height];
}

Matrix::Matrix(const Matrix& m)
{
	width = m.width;
	height = m.height;
	data = new cl_float[width*height];
	memcpy(data, m.data, width*height*sizeof(cl_float));
}

Matrix::~Matrix()
{
	delete [] data;
}

Matrix& Matrix::operator=(const Matrix& m)
{
	if (this!=&m)
	{
		delete [] data;
		width = m.width;
		height = m.height;
		data = new cl_float[width*height];
		memcpy(data, m.data, width*height*sizeof(cl_float));
	}
	return *this;
}

Matrix Matrix::operator+(const Matrix& m)
{
	if (m.width==width && m.height==height)
	{
		Matrix result(width, height);

		if (!useGPU)
		{
			// use CPU
			for (int i=0 ; i<width*height ; i++)
				result.data[i]=data[i]+m.data[i];
			return result;
		}
		else	// use GPU
		{
			cl_int status;

			// create buffers
			cl_mem Buffer = clCreateBuffer(OpenCLmgr->context, CL_MEM_READ_ONLY, width*height*sizeof(cl_float), NULL, NULL);
			status = clEnqueueWriteBuffer(OpenCLmgr->commandQueue, Buffer, CL_TRUE, 0, width*height*sizeof(cl_float), data, 0, NULL, NULL);
			//CHECK_SUCCESS("Error: writing buffer!")

			cl_mem BBuffer = clCreateBuffer(OpenCLmgr->context, CL_MEM_READ_ONLY, m.width*m.height*sizeof(cl_float), NULL, NULL);
			status = clEnqueueWriteBuffer(OpenCLmgr->commandQueue, BBuffer, CL_TRUE, 0, m.width*m.height*sizeof(cl_float), m.data, 0, NULL, NULL);
			//CHECK_SUCCESS("Error: writing buffer!")

			cl_mem CBuffer = clCreateBuffer(OpenCLmgr->context, CL_MEM_WRITE_ONLY , result.width*result.height*sizeof(cl_float), NULL, NULL);

			// Set kernel arguments.
			status = clSetKernelArg(OpenCLmgr->matadd_kernel, 0, sizeof(cl_int), (void *)&width);
			status |= clSetKernelArg(OpenCLmgr->matadd_kernel, 1, sizeof(cl_int), (void *)&height);
			status |= clSetKernelArg(OpenCLmgr->matadd_kernel, 2, sizeof(cl_mem), (void *)&Buffer);
			status |= clSetKernelArg(OpenCLmgr->matadd_kernel, 3, sizeof(cl_int), (void *)&m.width);
			status |= clSetKernelArg(OpenCLmgr->matadd_kernel, 4, sizeof(cl_int), (void *)&m.height);
			status |= clSetKernelArg(OpenCLmgr->matadd_kernel, 5, sizeof(cl_mem), (void *)&BBuffer);
			status |= clSetKernelArg(OpenCLmgr->matadd_kernel, 6, sizeof(cl_int), (void *)&result.width);
			status |= clSetKernelArg(OpenCLmgr->matadd_kernel, 7, sizeof(cl_int), (void *)&result.height);
			status |= clSetKernelArg(OpenCLmgr->matadd_kernel, 8, sizeof(cl_mem), (void *)&CBuffer);
			//CHECK_SUCCESS("Error: setting kernel argument!")
	
			// Run the kernel.
			size_t global_work_size[2] = {result.width, result.height};
			size_t local_work_size[2] = {result.width, result.height};
			status = clEnqueueNDRangeKernel(OpenCLmgr->commandQueue, OpenCLmgr->matadd_kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
			//CHECK_SUCCESS("Error: enqueuing kernel!")

			// Read the output back to host memory.
			status = clEnqueueReadBuffer(OpenCLmgr->commandQueue, CBuffer, CL_TRUE, 0, result.width*result.height*sizeof(cl_float), result.data, 0, NULL, NULL);
			//CHECK_SUCCESS("Error: reading buffer!")

			// release buffers
			status = clReleaseMemObject(Buffer);		
			status = clReleaseMemObject(BBuffer);
			status = clReleaseMemObject(CBuffer);

			return result;
		}
	}
	else
	{
		return Matrix();
	}
}

cl_float& Matrix::operator[](int index)
{
	if (index>=0 && index<width*height)
	{
		return data[index];
	}
	else
	{
		return dummy;
	}
}
	

cl_float& Matrix::Elem(int ix, int iy)
{
	int index = INDEX(*this, ix, iy);
	return (*this)[index];
}
