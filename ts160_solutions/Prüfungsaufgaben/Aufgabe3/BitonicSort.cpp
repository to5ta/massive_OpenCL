#include "BitonicSort.h"
#include <cstring>
#include "../shared/clstatushelper.h"

OpenCLMgr* BitonicSort::OpenCLmgr = NULL;


BitonicSort::BitonicSort(){

    OpenCLmgr = new OpenCLMgr();
    OpenCLmgr->buildProgram("../bitonic.cl");
    const char * kernel_names[] = {"bitonic_kernel"};
    OpenCLmgr->createKernels(kernel_names, 1);

}



BitonicSort::~BitonicSort(){
    delete OpenCLmgr;
    delete data;
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

    cl_int status=0;

    cl_mem InBuffer = clCreateBuffer(this->OpenCLmgr->context, CL_MEM_READ_ONLY, datalength*sizeof(cl_uint), NULL, NULL);

    status          = clEnqueueWriteBuffer(OpenCLmgr->commandQueue, InBuffer, CL_TRUE, 0, this->datalength*sizeof(cl_uint), this->data, 0, NULL, NULL);
    check_error(status);

    cl_mem OutBuffer = clCreateBuffer(OpenCLmgr->context, CL_MEM_WRITE_ONLY , this->datalength*sizeof(cl_uint), NULL, NULL);

    // set arguments here....
    status  = clSetKernelArg(OpenCLmgr->kernels["bitonic_kernel"], 0, sizeof(cl_int), (void *)&datalength);
    status  = clSetKernelArg(OpenCLmgr->kernels["bitonic_kernel"], 1, sizeof(cl_mem), (void *)&InBuffer);
    status |= clSetKernelArg(OpenCLmgr->kernels["bitonic_kernel"], 2, sizeof(cl_mem), (void *)&OutBuffer);
    check_error(status);

    // Run the kernel.
    size_t gws_0 = ((datalength-1)/16+1)*16;
//    size_t global_work_size[1] = {gws_0};
    size_t global_work_size[1] = {1024};
    size_t local_work_size[1] = {1024};


    printf("Global Work Size: [%i], ", global_work_size[0]);
    printf("Local Work Size:  [%i]\n", local_work_size[0]);

    // actually start kernel ("enqueue")
    status = clEnqueueNDRangeKernel(OpenCLmgr->commandQueue, OpenCLmgr->kernels["bitonic_kernel"], 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
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


