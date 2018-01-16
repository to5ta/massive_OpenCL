#include "PrefixSum.h"
#include <cmath>
#include <cstring>
#include "../shared/clstatushelper.h"
#include "../shared/ansi_colors.h"
#include <assert.h>


OpenCLMgr *PrefixSum::OpenCLmgr = NULL;


PrefixSum::PrefixSum() {
    OpenCLmgr = new OpenCLMgr();
    OpenCLmgr->buildProgram("../prefixSum.cl");
    const char *kernel_names[] = {"prefixSum_kernel"};
    OpenCLmgr->createKernels(kernel_names, 1);

}


PrefixSum::~PrefixSum() {
    delete OpenCLmgr;
}


void PrefixSum::loadData(int newdatalength, cl_uint *newdata) {
    if (this->inputData != NULL) {
        delete[] this->inputData;
        inputData = NULL;

        delete[] this->prefixData;
        prefixData = NULL;
    }

    int multiplelength = ((newdatalength - 1) / 256 + 1) * 256;

    this->reallength = newdatalength;
    this->datalength = multiplelength;

    this->inputData = new cl_uint[multiplelength]();
    memcpy(this->inputData, newdata, sizeof(cl_uint) * newdatalength);

    this->prefixData = new cl_uint[multiplelength]();

    printInfo();
}


void PrefixSum::plotData(cl_uint *owndata, int length) {
    int columns = 32;
    int lines = 3;
    printf("[");
    for (int i = 0; i < length; i++) {
        if (i < reallength)
            printf("%3i", owndata[i]);
        else {
            printf(ANSI_COLOR_RED);
            printf("%3i", owndata[i]);
            printf(ANSI_COLOR_RESET);
        }

        if (i < length - 1)
            printf(", ");
        else
            printf("  ]");
        if ((i + 1) % columns == 0)
            printf("\n ");

        if (i + 1 == columns * lines && length > columns * lines * 2 + 1) {
            printf("  ...\n ");
            i = length - 1 - columns * lines;
        }
    }
    printf("\n");
}


void PrefixSum::printInfo() {

    printf("Data Buffer Length: %5i\n", datalength);
    printf("Real Data Length:   %5i, %i dummy elements at the end.\n", reallength, datalength - reallength);

    plotData(this->inputData, datalength);
//
//    printf("Prefix Data\n");
//    plotData(this->prefixData);

}


//void PrefixSum::prefixSumsPerBlock(cl_uint *inputdata, cl_uint *blockwiseprefix, int length) {
void PrefixSum::prefixSumsPerBlock(cl_mem input, cl_mem prefix, cl_mem helpsum, int length) {
//    printf("Inputdata adress: %x\n", inputdata);


//    start kernels here

//    kernel should first calculate prefixes per block, storing it in prefix

//    sum then last element and last prefix sum per block, gives sum of a block

    // how do we get the data back?
    // do we have to?
    // is it still there?
    // do we need write/read/write-read buffers?!?!?

    return;


    cl_int status = 0;

    cl_mem BufferA = clCreateBuffer(this->OpenCLmgr->context, CL_MEM_READ_ONLY, datalength * sizeof(cl_uint), NULL,
                                    NULL);

    status = clEnqueueWriteBuffer(OpenCLmgr->commandQueue, BufferA, CL_TRUE, 0, this->datalength * sizeof(cl_uint),
                                  this->inputData, 0, NULL, NULL);
    check_error(status);

    cl_mem OutBuffer = clCreateBuffer(OpenCLmgr->context, CL_MEM_WRITE_ONLY, this->datalength * sizeof(cl_uint), NULL,
                                      NULL);


    // Run the kernel.
    size_t gws_0 = (((datalength) - 1) / 16 + 1) * 16;
//    size_t global_work_size[1] = {gws_0};
    size_t global_work_size[1] = {gws_0};
    size_t local_work_size[1] = {gws_0};

    // set arguments here....
    status = clSetKernelArg(OpenCLmgr->kernels["bitonic_kernel"], 0, sizeof(cl_int), (void *) &gws_0);
    status = clSetKernelArg(OpenCLmgr->kernels["bitonic_kernel"], 1, sizeof(cl_mem), (void *) &BufferA);
    status |= clSetKernelArg(OpenCLmgr->kernels["bitonic_kernel"], 2, sizeof(cl_mem), (void *) &OutBuffer);
    check_error(status);


    printf("Global Work Size: [%5i], ", global_work_size[0]);
    printf("Local Work Size:  [%5i]\n", local_work_size[0]);

    // actually start kernel ("enqueue")
    status = clEnqueueNDRangeKernel(OpenCLmgr->commandQueue, OpenCLmgr->kernels["bitonic_kernel"], 1, NULL,
                                    global_work_size, local_work_size, 0, NULL, NULL);
    check_error(status);

    // Read the output back to host memory.
    status = clEnqueueReadBuffer(OpenCLmgr->commandQueue, OutBuffer, CL_TRUE, 0, datalength * sizeof(cl_uint),
                                 inputData, 0, NULL, NULL);
    check_error(status);

    // cut leading Zeros again!
    cl_uint *raw_data = new cl_uint[reallength];
    memcpy(raw_data, this->inputData + (datalength - reallength), sizeof(cl_uint) * reallength);
    delete[] this->inputData;
    this->inputData = NULL;
    this->inputData = raw_data;

    // release buffers
    status = clReleaseMemObject(BufferA);
    status |= clReleaseMemObject(OutBuffer);
    check_error(status);
}


void PrefixSum::prefixSumGPU() {

    int levels = (int) (log(datalength) / log(256));
    cl_uint *inputs[levels];
    cl_uint *prefix[levels];

    printf("Start Prefix calculation...\n");

    printf("Real Data Length: %12i\n", reallength);
    printf("Round up to x256: %12i\n", datalength);

    printf("Calculation will imply %i Levels!\n", levels);

    if (datalength * 4 > OpenCLmgr->maxMem) {
        printf(ANSI_COLOR_RED);
        printf("Sorry, you want to allocate %i Bytes but the card does only support %i Bytes at once!", datalength * 4,
               OpenCLmgr->maxMem);
        printf("Aborting....");
        printf(ANSI_COLOR_RESET);
        return;
    } else {
        printf("Largest Single Buffer:  %12i Bytes.\n", datalength * 4);
        printf("Card supports up to:    %12u Bytes at once!\n", OpenCLmgr->maxMem);
    }

    // dangerous, ensure this data is never freed
    inputs[0] = this->inputData;
    prefix[0] = this->prefixData;

    int bufferLevelSize = datalength;

    // create level buffers, level 0 buffer given through class member buffers
    for (int level = 1; level < levels; level++) {

        bufferLevelSize = ((bufferLevelSize - 1) / 256 + 1);
        bufferLevelSize = ((bufferLevelSize - 1) / 256 + 1) * 256;
        printf("Creating Buffers of Length %5i for Level %i...\n", bufferLevelSize, level);

        inputs[level] = (cl_uint *) (malloc(sizeof(cl_uint) * bufferLevelSize));
//        inputs[level] = new cl_uint[bufferLevelSize]();
        assert(inputs[level] != NULL);
        prefix[level] = (cl_uint *) (malloc(sizeof(cl_uint) * bufferLevelSize));
//        inputs[level] = new cl_uint[bufferLevelSize]();
        assert(prefix[level] != NULL);
    }


    // do the blockwise prefix stuff repetitive
    for (int level = 0; level < levels; level++) {
        printf("Calculate PrefixSums block by block for Level %i...\n", level);

//        clCreateBuffer(....

//        prefixSumsPerBlock(....

//        addPrefixAndBlock(.....  block + blocksum-buffer  
     }


    // build real prefixes, add previous prefix sums (block by block internal + block sum)
    for (int output = levels; output >= 0; output--) {

    }


    for (int level = 1; level < levels; level++) {
        printf("Free Buffers for Level %i...\n", level);
//        delete [] inputs[level];
        free(inputs[level]);
//        delete [] prefix[level];
        free(prefix[level]);
    }

    return;


}