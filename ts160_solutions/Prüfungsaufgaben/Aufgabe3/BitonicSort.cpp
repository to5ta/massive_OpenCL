#include "BitonicSort.h"
#include <cstring>
#include "../shared/clstatushelper.h"

OpenCLMgr* BitonicSort::OpenCLmgr = NULL;


BitonicSort::BitonicSort(){

    OpenCLmgr->buildProgram("bitonic.cl");

    const char * kernel_names[] = {"bitonic_kernel"};
    OpenCLmgr->createKernels(kernel_names, 1);

}



BitonicSort::~BitonicSort(){

}


void BitonicSort::loadData(int newdatalength, cl_uint *newdata){

    if(this->data!=NULL) {
        delete [] this->data;
    }

    this->datalength = newdatalength;
    this->data = new cl_uint[newdatalength];
    memcpy(this->data, newdata, sizeof(cl_uint)*newdatalength);
}



void BitonicSort::sortGPU(){

    cl_int status;
    cl_mem InBuffer = clCreateBuffer(this->OpenCLmgr->context, CL_MEM_READ_ONLY, this->datalength*sizeof(cl_uint), NULL, NULL);
    status          = clEnqueueWriteBuffer(OpenCLmgr->commandQueue, InBuffer, CL_TRUE, 0, this->datalength*sizeof(cl_uint), this->data, 0, NULL, NULL);
    check_error(status);
    cl_mem OutBuffer = clCreateBuffer(OpenCLmgr->context, CL_MEM_WRITE_ONLY , this->datalength*sizeof(cl_uint), NULL, NULL);
    check_error(status);

    // set arguments here....
//    status = clSetKernelArg(OpenCLmgr->summe_kernel, 0, sizeof(cl_mem), (void *)&InBuffer);
    check_error(status);
//    status = clSetKernelArg(OpenCLmgr->summe_kernel, 1, sizeof(cl_mem), (void *)&OutBuffer);
    status = clSetKernelArg(OpenCLmgr->kernels["bitonic_sort"], 1, sizeof(cl_mem), (void *)&OutBuffer);

    check_error(status);

    // Run the kernel.
    size_t global_work_size[1] = {1};
    size_t local_work_size[2] = {datalength};

    // actually start kernel ("enqueue")
    status = clEnqueueNDRangeKernel(OpenCLmgr->commandQueue, OpenCLmgr->kernels["bitonic_sort"], 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    check_error(status);

    // Read the output back to host memory.
    status = clEnqueueReadBuffer(OpenCLmgr->commandQueue, OutBuffer, CL_TRUE, 0, datalength*sizeof(cl_uint), data, 0, NULL, NULL);
    check_error(status);

    // release buffers
    status = clReleaseMemObject(InBuffer);
    status |= clReleaseMemObject(OutBuffer);
    check_error(status);
}

void BitonicSort::sortCPU(){

}


