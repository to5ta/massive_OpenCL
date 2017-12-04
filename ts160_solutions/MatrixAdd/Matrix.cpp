
#include <stdlib.h>
#include <string.h>
#include <cmath>

using namespace std;

#include "Matrix.h"
#include "OpenCLMgr.h"

#define INDEX(M,x,y) ((y)*((M).width)+(x))

cl_float Matrix::dummy;
int Matrix::useGPU = 0;
int Matrix::useSharedMemory=0;
OpenCLMgr* Matrix::OpenCLmgr = NULL;;


Matrix::Matrix()
{ 
	width = height = 0;
	data = NULL;
    width = 0;
    height = 0;
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

Matrix::Matrix(int w, int h, float *init_data) {
	width = w;
	height = h;
	data = new cl_float[width*height];

	memcpy(data, init_data, width*height*sizeof(cl_float));
}


Matrix::Matrix(int w, int h, int TYPE) {
	width = w;
	height = h;
	if(TYPE==MATRIX_NEW_ZEROS) {
		data = new cl_float[width*height];
		for(int i=0; i<w*h; i++){
			data[i] = 0.f;
		}

	}
	if(TYPE==MATRIX_NEW_RANDOM) {
		data = new cl_float[width*height];
		for(int i=0; i<w*h; i++){
			data[i] = (cl_float)rand()/RAND_MAX;
		}
	}
	if(TYPE==MATRIX_NEW_ONES) {
		data = new cl_float[width*height];
		for(int i=0; i<w*h; i++){
			data[i] = 1.f;
		}
	}
	if(TYPE==MATRIX_NEW_IDENTITY) {
		data = new cl_float[width*height];
		for(int i=0; i<w*h; i++){
			if(i/w==i%w)
				data[i] = 1.f;
			else
				data[i] = 0.f;
		}
	}

	// memcpy(data, init_data, width*height*sizeof(cl_float));
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

int Matrix::operator==(const Matrix& m)
{
	if(width!=m.width or height!=m.height)
		return 0;

	else {
		for(int i=0; i<m.height*m.width; i++){
			if(m.data[i]!=this->data[i]){
				cl_float diff = abs(m.data[i]-this->data[i]);
				if(diff>TOLERANCE){
					printf("ERROR ID %i: DIFF: %f  '%s': %f  '%s': %f\n", i, diff, m.name.c_str(), m.data[i], this->name.c_str(), this->data[i]);
					return 0;
				} else if(WARN_TOLERANCE) {
					printf("WARNING ID %i: DIFF: %f = abs(%f-%f)\n", i, diff, m.data[i], this->data[i]);
//					printf("WARNING: Index %i: abs(%f-%f)=%f\n", i, m.data[i], this->data[i], diff);
				}
			}
		}
		return 1;
	}


//	if(memcmp(data, m.data, width*height*sizeof(cl_float))!=0)
//		return 0;
//	else
//		return 1;

//	for(int i=0; i<width*height; i++){
//		if(m.data[i]!=data[i])
//			printf("%i: %f(own) != %f",i, data[i], m.data[i]);
//			return false;
//	}
//	return true;
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

Matrix Matrix::operator*(const Matrix& m)
{
	timingBegin();
    if (width==m.height)
    {
//        Matrix result(m.width, height);
        Matrix result(m.width, height, MATRIX_NEW_ZEROS);

        if (!useGPU)
        {
			 printf("\nUsing CPU..\n");
            // row, column, incrementing index in sum
            int r,c, index;
            cl_float sum, _a, _b, temp;

            for(r=0; r<result.height; r++){
                for(c=0; c<result.width; c++){
                    sum = 0;
                    for(index=0; index<width; index++) {
						_a = this->data[index + r * width];
						_b = m.data[c + m.width * index];
						temp = _a * _b;
						// printf("A[%i,%i](%2.1f)*B[%i,%i](%2.1f)=(%2.2f) + ", index, r, _a, c, index, _b, temp);
						sum += temp;
					}
					// printf("SUM  %3.1f\n", sum);
					result.data[c + r * result.width] = sum;
                }
            }
			timingEnd();
            return result;
        }
        else if(useSharedMemory && useGPU){

			printf("\nUsing GPU (SharedMemory)..\n");
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
			status = clSetKernelArg(OpenCLmgr->matmulshared_kernel,  0, sizeof(cl_int), (void *)&width);
			status |= clSetKernelArg(OpenCLmgr->matmulshared_kernel, 1, sizeof(cl_int), (void *)&height);
			status |= clSetKernelArg(OpenCLmgr->matmulshared_kernel, 2, sizeof(cl_mem), (void *)&Buffer);

			status |= clSetKernelArg(OpenCLmgr->matmulshared_kernel, 3, sizeof(cl_int), (void *)&m.width);
			status |= clSetKernelArg(OpenCLmgr->matmulshared_kernel, 4, sizeof(cl_int), (void *)&m.height);
			status |= clSetKernelArg(OpenCLmgr->matmulshared_kernel, 5, sizeof(cl_mem), (void *)&BBuffer);

			status |= clSetKernelArg(OpenCLmgr->matmulshared_kernel, 6, sizeof(cl_int), (void *)&result.width);
			status |= clSetKernelArg(OpenCLmgr->matmulshared_kernel, 7, sizeof(cl_int), (void *)&result.height);
			status |= clSetKernelArg(OpenCLmgr->matmulshared_kernel, 8, sizeof(cl_mem), (void *)&CBuffer);
			//CHECK_SUCCESS("Error: setting kernel argument!")
//			printf("status %i\n", status);


//			printf("Result.Wi")
			size_t gws_0 = ((result.width-1)/16+1)*16;
			size_t gws_1 = ((result.height-1)/16+1)*16;

			size_t global_work_size[2] = {gws_0, gws_1};
			size_t local_work_size[2] = {16, 16};

			printf("Global Work Size: [%i,%i], ", global_work_size[0], global_work_size[1]);
			printf("Local Work Size:  [%i,%i]\n", local_work_size[0], local_work_size[1]);

			status = clEnqueueNDRangeKernel(OpenCLmgr->commandQueue, OpenCLmgr->matmulshared_kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
			status = clEnqueueNDRangeKernel(OpenCLmgr->commandQueue, OpenCLmgr->matmulshared_kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
//			printf("status %i\n", status);
            // CHECK_SUCCESS("Error: enqueuing kernel!")

			// Read the output back to host memory.
			status = clEnqueueReadBuffer(OpenCLmgr->commandQueue, CBuffer, CL_TRUE, 0, result.width*result.height*sizeof(cl_float), result.data, 0, NULL, NULL);
			//CHECK_SUCCESS("Error: reading buffer!")

			// release buffers
			status = clReleaseMemObject(Buffer);
			status = clReleaseMemObject(BBuffer);
			status = clReleaseMemObject(CBuffer);
			timingEnd();
			return result;
        }


        else if(useGPU && !useSharedMemory)	// use GPU
        {
			printf("\nUsing GPU..\n");
            cl_int status;


            printf("About to allocate: %i bytes\n", width*height*sizeof(cl_float));
            printf("About to allocate: %f MB\n", (width*height*sizeof(cl_float))/1048576.f);
            // create buffers
            cl_mem Buffer = clCreateBuffer(OpenCLmgr->context, CL_MEM_READ_ONLY, width*height*sizeof(cl_float), NULL, NULL);
            status = clEnqueueWriteBuffer(OpenCLmgr->commandQueue, Buffer, CL_TRUE, 0, width*height*sizeof(cl_float), data, 0, NULL, NULL);
            //CHECK_SUCCESS("Error: writing buffer!")

            cl_mem BBuffer = clCreateBuffer(OpenCLmgr->context, CL_MEM_READ_ONLY, m.width*m.height*sizeof(cl_float), NULL, NULL);
            status = clEnqueueWriteBuffer(OpenCLmgr->commandQueue, BBuffer, CL_TRUE, 0, m.width*m.height*sizeof(cl_float), m.data, 0, NULL, NULL);
            //CHECK_SUCCESS("Error: writing buffer!")

            cl_mem CBuffer = clCreateBuffer(OpenCLmgr->context, CL_MEM_WRITE_ONLY , result.width*result.height*sizeof(cl_float), NULL, NULL);

            // Set kernel arguments.
            status = clSetKernelArg(OpenCLmgr->matmul_kernel,  0, sizeof(cl_int), (void *)&width);
            status |= clSetKernelArg(OpenCLmgr->matmul_kernel, 1, sizeof(cl_int), (void *)&height);
            status |= clSetKernelArg(OpenCLmgr->matmul_kernel, 2, sizeof(cl_mem), (void *)&Buffer);

            status |= clSetKernelArg(OpenCLmgr->matmul_kernel, 3, sizeof(cl_int), (void *)&m.width);
            status |= clSetKernelArg(OpenCLmgr->matmul_kernel, 4, sizeof(cl_int), (void *)&m.height);
            status |= clSetKernelArg(OpenCLmgr->matmul_kernel, 5, sizeof(cl_mem), (void *)&BBuffer);

            status |= clSetKernelArg(OpenCLmgr->matmul_kernel, 6, sizeof(cl_int), (void *)&result.width);
            status |= clSetKernelArg(OpenCLmgr->matmul_kernel, 7, sizeof(cl_int), (void *)&result.height);
            status |= clSetKernelArg(OpenCLmgr->matmul_kernel, 8, sizeof(cl_mem), (void *)&CBuffer);
//            CHECK_SUCCESS("Error: setting kernel argument!")


//			// Default Code
			size_t global_work_size[2] = {result.width, result.height};
			size_t local_work_size[2] = {result.width, result.height};

            status = clEnqueueNDRangeKernel(OpenCLmgr->commandQueue, OpenCLmgr->matmul_kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
            // CHECK_SUCCESS("Error: enqueuing kernel!")

            // Read the output back to host memory.
            status = clEnqueueReadBuffer(OpenCLmgr->commandQueue, CBuffer, CL_TRUE, 0, result.width*result.height*sizeof(cl_float), result.data, 0, NULL, NULL);
            //CHECK_SUCCESS("Error: reading buffer!")

            // release buffers
            status = clReleaseMemObject(Buffer);
            status = clReleaseMemObject(BBuffer);
            status = clReleaseMemObject(CBuffer);
			timingEnd();
            return result;
        } else {
			printf("Wrong Parameters: \nuseSharedMemory: %i\nuseGPU: %i", useSharedMemory, useGPU);
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

void Matrix::plot(void)
{
    int x = 0;
    int y = 0;

	for (x=0 ; x<this->height ; x++)
	{
		if(x==0)
			printf("[[");

		for (y=0 ; y<this->width; y++){
            if(y==0 and x!=0)
                printf(" [");

			printf("%7.3f",this->Elem(x,y));
			if(y+2<=this->width)
				printf(", ");
            else if(x+1==this->height)
                printf("]");
            else
                printf("]\n");
		}
	}
    printf("]\n");
}