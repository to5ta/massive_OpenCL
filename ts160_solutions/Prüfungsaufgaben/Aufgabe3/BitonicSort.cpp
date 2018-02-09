#include "BitonicSort.h"
#include <cstring>
#include <cmath>
#include "../shared/clstatushelper.h"

OpenCLMgr* BitonicSort::OpenCLmgr = NULL;


BitonicSort::BitonicSort(){

    OpenCLmgr = new OpenCLMgr( CL_QUEUE_PROFILING_ENABLE );
    OpenCLmgr->loadFile("../bitonic.cl");
    OpenCLmgr->buildProgram();
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
        data = NULL;
    }

//    if(pow( (int)(log2(newdatalength))+1

    int powerlength = (int)(pow(2, ceil(log2(newdatalength))));

    printf("Suggested Length: %i\n", powerlength);

    this->reallength = newdatalength;
    this->datalength = powerlength;
    this->data = new cl_uint[powerlength]();
    memcpy(this->data, newdata, sizeof(cl_uint)*newdatalength);
}



void BitonicSort::sortGPU(){

    cl_int status=0;

    cl_mem InBuffer = clCreateBuffer(this->OpenCLmgr->context, CL_MEM_READ_ONLY, datalength*sizeof(cl_uint), NULL, NULL);
    status          = clEnqueueWriteBuffer(OpenCLmgr->commandQueue, InBuffer, CL_TRUE, 0, this->datalength*sizeof(cl_uint), this->data, 0, NULL, NULL);
    check_error(status);

    cl_mem OutBuffer = clCreateBuffer(OpenCLmgr->context, CL_MEM_WRITE_ONLY , this->datalength*sizeof(cl_uint), NULL, NULL);


    // Run the kernel.
    size_t gws_0 = (((datalength)-1)/16+1)*16;
//    size_t global_work_size[1] = {gws_0};
    size_t global_work_size[1] = {gws_0};
    size_t local_work_size[1] = {gws_0};

    // set arguments here....
    status  = clSetKernelArg(OpenCLmgr->kernels["bitonic_kernel"], 0, sizeof(cl_int), (void *)&gws_0);
    status  = clSetKernelArg(OpenCLmgr->kernels["bitonic_kernel"], 1, sizeof(cl_mem), (void *)&InBuffer);
    status |= clSetKernelArg(OpenCLmgr->kernels["bitonic_kernel"], 2, sizeof(cl_mem), (void *)&OutBuffer);
    check_error(status);


    printf("Global Work Size: [%i], ", global_work_size[0]);
    printf("Local Work Size:  [%i]\n", local_work_size[0]);

    // actually start kernel ("enqueue")
    status = clEnqueueNDRangeKernel(OpenCLmgr->commandQueue, OpenCLmgr->kernels["bitonic_kernel"], 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    check_error(status);

    // Read the output back to host memory.
    status = clEnqueueReadBuffer(OpenCLmgr->commandQueue, OutBuffer, CL_TRUE, 0, datalength*sizeof(cl_uint), data, 0, NULL, NULL);
    check_error(status);

    // cut leading Zeros again!
    cl_uint * raw_data = new cl_uint[reallength];
    memcpy(raw_data, this->data+(datalength-reallength), sizeof(cl_uint)*reallength);
    delete [] this->data;
    this->data = NULL;
    this->data = raw_data;

    // release buffers
    status = clReleaseMemObject(InBuffer);
    status |= clReleaseMemObject(OutBuffer);
    check_error(status);
}

void BitonicSort::sortCPU(){

}


