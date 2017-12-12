
#include <stdlib.h>
#include <string.h>

using namespace std;

#include "Matrix.h"
#include "OpenCLMgr.h"

#define INDEX(M,x,y) ((y)*((M).width)+(x))

cl_float Matrix::dummy;
int Matrix::useGPU = 1;
OpenCLMgr* Matrix::OpenCLmgr = NULL;


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
		else
        {
            cl_int status = 0;
            OpenCLMgr ClManager;
            auto context = ClManager.context;
//            auto program = ClManager.program;
            auto commandQueue = ClManager.commandQueue;
            
            /*Step 7: Initial input,output for the host and create memory objects for the kernel*/
            int Awidth = width;
            int Aheight = height;
            float* Aelements = data;
            int Bwidth = width;
            int Bheight = height;
            float* Belements = m.data;

            
            size_t NumElems = width * height;
            
            float *output = (float*) malloc(NumElems);
            
            cl_mem InAwidth = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(int), (void *) Awidth, NULL);
            cl_mem InAheight = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(int), (void *) Aheight, NULL);
            cl_mem InAelements = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, NumElems * sizeof(cl_float), (void *) Aelements, NULL);
            
            cl_mem InBwidth = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(int), (void *) Bwidth, NULL);
            cl_mem InBheight = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(int), (void *) Bheight, NULL);
            cl_mem InBelements = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, NumElems * sizeof(cl_float), (void *) Belements, NULL);
            
            cl_mem OutCwidth = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int), NULL, NULL);
            cl_mem OutCheight = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int), NULL, NULL);
            cl_mem OutCelements = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (NumElems * sizeof(cl_float)), NULL, NULL);
            
            
            /*Step 8: Create kernel object */
            cl_kernel kernel = ClManager.matadd_kernel;
            
            /*Step 9: Sets Kernel arguments.*/
            status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&InAwidth);
            status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&InAheight);
            status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&InAelements);
            status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&InBwidth);
            status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&InBheight);
            status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&InBelements);
            status = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&OutCwidth);
            status = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&OutCheight);
            status = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *)&OutCelements);
            
            /*Step 10: Running the kernel.*/
            size_t global_work_size[1] = {NumElems};
            size_t local_work_size[1] = {NumElems};
            status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
            
            /*Step 11: Read the cout put back to host memory.*/
            status = clEnqueueReadBuffer(commandQueue, OutCelements, CL_TRUE, 0, NumElems * sizeof(cl_float), output, 0, NULL, NULL);
            
            for (int i=0 ; i<width*height ; i++){
                cout << output[i] << endl;
                result.data[i]= output[i];
            }
            
            /*Step 12: Clean the resources.*/
//            status = clReleaseKernel(kernel);                //Release kernel.
//            status = clReleaseProgram(program);                //Release the program object.
            
            status = clReleaseMemObject(InAwidth);        //Release mem object.
            status = clReleaseMemObject(InAheight);
            status = clReleaseMemObject(InAelements);        //Release mem object.
            status = clReleaseMemObject(InBwidth);
            status = clReleaseMemObject(InBheight);        //Release mem object.
            status = clReleaseMemObject(InBelements);
            status = clReleaseMemObject(OutCwidth);        //Release mem object.
            status = clReleaseMemObject(OutCheight);
            status = clReleaseMemObject(OutCelements);        //Release mem object.
            
//            status = clReleaseCommandQueue(commandQueue);    //Release  Command queue.
//            status = clReleaseContext(context);                //Release context.
            
            
            if (output != NULL)
            {
                free(output);
                output = NULL;
            }
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
