#include "BitonicSort.h"
#include <cstring>
#include <cmath>
#include "../shared/clstatushelper.h"
#include "../shared/ansi_colors.h"
#include <stdlib.h>

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
    delete [] gpu_data;
    delete [] cpu_data;
}


void BitonicSort::loadData(int newdatalength, cl_uint *newdata){

    if(this->gpu_data!=NULL) {
        delete [] this->gpu_data;
        gpu_data = NULL;
    }

    if(this->cpu_data!=NULL) {
        delete [] this->cpu_data;
        cpu_data = NULL;
    }

//    if(pow( (int)(log2(newdatalength))+1

    int powerlength = (int)(pow(2, ceil(log2(newdatalength))));

    printf("Suggested Length: %i\n", powerlength);

    this->reallength    = newdatalength;
    this->datalength    = powerlength;
    this->gpu_data      = new cl_uint[powerlength]();
    this->cpu_data      = new cl_uint[newdatalength]();
    memcpy(this->gpu_data, newdata, sizeof(cl_uint)*newdatalength);
    memcpy(this->cpu_data, newdata, sizeof(cl_uint)*newdatalength);
}



int compare (const void * a, const void * b)
{
    return ( *(uint*)a - *(uint*)b );
}


void BitonicSort::sortCPU() {
    qsort (cpu_data, reallength, sizeof(uint), compare);
}


void BitonicSort::printData(uint* data,
                            uint length,
                            uint margin){

    printf("-");
    char skip = 0;
    for(int i=0; i<length; i++){
        printf("%3i", i);

        if(i>margin && !skip && margin*2<length){
            printf(",  ...  ");
            i=length-margin-1;
            skip=1;
            continue;
        }

        if(i<length-1) {
            printf(", ");
        }
    }
    printf("-\n");

    skip=0;

    printf("[");
    for(int i=0; i<length; i++){
        printf("%3i", data[i]);

        if(i>margin && !skip && margin*2<length){
            printf(",  ...  ");
            i=length-margin-1;
            skip=1;
            continue;
        }

        if(i<length-1) {
            printf(", ");
        }
    }
    printf("]\n");
}




void BitonicSort::sortGPU(){

    cl_int status=0;


    cl_mem Buffer = clCreateBuffer(OpenCLmgr->context,
                                      CL_MEM_READ_WRITE,
                                      this->datalength*sizeof(cl_uint),
                                      NULL,
                                      NULL);
    check_error(status);

    status          = clEnqueueWriteBuffer(OpenCLmgr->commandQueue,
                                           Buffer,
                                           CL_TRUE,
                                           0,
                                           this->datalength*sizeof(cl_uint),
                                           this->gpu_data,
                                           0,
                                           NULL,
                                           NULL);
    check_error(status);

    // Run the kernel.
    size_t gws_0 = ((((datalength)-1)/16+1)*16) / 2;
//    size_t global_work_size[1] = {gws_0};

    // gws is a power of 2



    size_t global_work_size[1] = {gws_0};
    size_t local_work_size[1]  = {gws_0};

    if(gws_0<=512){
        local_work_size[0]  = gws_0;
    } else {
        local_work_size[0]  = 512;
    }



    // set arguments here....
    status  = clSetKernelArg(OpenCLmgr->kernels["bitonic_kernel"],
                             0,
                             sizeof(cl_int),
                             (void *)&gws_0);

    status |= clSetKernelArg(OpenCLmgr->kernels["bitonic_kernel"],
                             1,
                             sizeof(cl_mem),
                             (void *)&Buffer);
    check_error(status);


    printf("Global Work Size: [%i]\n", global_work_size[0]);
    printf("Local Work Size:  [%i]\n", local_work_size[0]);

    printf("GPU Buffer: 0x%x\n", Buffer);


    // actually start kernel ("enqueue")
    status = clEnqueueNDRangeKernel(OpenCLmgr->commandQueue,
                                    OpenCLmgr->kernels["bitonic_kernel"],
                                    1,
                                    NULL,
                                    global_work_size,
                                    local_work_size,
                                    0,
                                    NULL,
                                    NULL);
    check_error(status);


    /// Read the output back to host memory.
    status = clEnqueueReadBuffer(OpenCLmgr->commandQueue,
                                 Buffer,
                                 CL_TRUE,
                                 0,
                                 this->datalength*sizeof(cl_uint),
                                 this->gpu_data,
                                 0,
                                 NULL,
                                 NULL);
    check_error(status);


    /// cut leading Zeros again!
     cl_uint * raw_data = new cl_uint[reallength]();
     memcpy(raw_data, this->gpu_data+(datalength-reallength), sizeof(cl_uint)*reallength);
     delete [] this->gpu_data;
     this->gpu_data = NULL;
     this->gpu_data = raw_data;

    // release buffers
    status = clReleaseMemObject(Buffer);
    check_error(status);
}

