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
    const char *kernel_names[] = {"addBlockSumToPrefix_kernel", "calcBlockSum_kernel", "prefixBlockwise_kernel"};
    OpenCLmgr->createKernels(kernel_names, 3);

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



void PrefixSum::addBlockSumToPrefix(cl_mem prefix0, cl_mem input1, int length1, int blocksize) {

}



void PrefixSum::calcBlockSums(cl_mem input0, cl_mem prefix0, cl_mem input1, int length1, int blocksize) {

}



void PrefixSum::prefixBlockwise(cl_mem input0, cl_mem prefix0, int length0, int blocksize) {
    cl_int status;
    status = clSetKernelArg(OpenCLmgr->kernels["prefixBlockwise_kernel"], 0, sizeof(cl_mem), (void *) &input0);
    check_error(status);

    status |= clSetKernelArg(OpenCLmgr->kernels["prefixBlockwise_kernel"], 1, sizeof(cl_mem), (void *) &prefix0);
    status |= clSetKernelArg(OpenCLmgr->kernels["prefixBlockwise_kernel"], 2, sizeof(cl_int), (void *) &length0);
    status |= clSetKernelArg(OpenCLmgr->kernels["prefixBlockwise_kernel"], 3, sizeof(cl_int), (void *) &blocksize);
    check_error(status);

    size_t gws[1] = {blocksize};
    size_t lws[1] = {blocksize};

    status = clEnqueueNDRangeKernel(OpenCLmgr->commandQueue, OpenCLmgr->kernels["prefixBlockwise_kernel"], 1, NULL, gws,
                                    lws, 0, NULL, NULL);
    check_error(status);

    return;
}



void PrefixSum::prefixSumGPU() {

    int blocksize = 256;

    cl_int status;

    int levels = (int) (log(datalength) / log(256));
    cl_mem inputs[levels];
    cl_mem prefix[levels];
    int lengths[levels];

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

    inputs[0] = clCreateBuffer(OpenCLmgr->context, CL_MEM_READ_ONLY, datalength * sizeof(cl_uint), NULL, NULL);
    status = clEnqueueWriteBuffer(OpenCLmgr->commandQueue, inputs[0], CL_TRUE, 0, datalength * sizeof(cl_uint),
                                  inputData, 0, NULL, NULL);
    check_error(status);
    prefix[0] = clCreateBuffer(OpenCLmgr->context, CL_MEM_READ_WRITE, datalength * sizeof(cl_uint), NULL, NULL);

    // buffer Length Size
    int bLS = datalength;
    lengths[0] = bLS;

    // create level buffers
    for (int level = 1; level < levels; level++) {

        bLS = ((bLS - 1) / 256 + 1);
        bLS = ((bLS - 1) / 256 + 1) * 256;
        lengths[level] = bLS;
        printf("Creating Buffers of Length %5i for Level %i...\n", bLS, level);

        inputs[level] = clCreateBuffer(OpenCLmgr->context, CL_MEM_READ_WRITE, bLS * sizeof(cl_uint), NULL, NULL);
        prefix[level] = clCreateBuffer(OpenCLmgr->context, CL_MEM_READ_WRITE, bLS * sizeof(cl_uint), NULL, NULL);
    }


    // do the blockwise prefix stuff repetitive
    for (int level = 0; level < levels; level++) {
        printf("Calculate PrefixSums block by block for Level %i...\n", level);

        if(level>0){
            // blocksum aka new input data
            calcBlockSums(inputs[level-1], prefix[level-1], inputs[level], lengths[level], blocksize);
        }

        // simple blockwise prefixes
        prefixBlockwise(inputs[level], prefix[level], lengths[level], blocksize);
    }


    // build real prefixes, add previous prefix sums (block by block internal + block sum)
    for (int output = levels; output > 1; output--) {
        // addBlockSumToPrefix
        printf("Add BlockSums to Blockwise Prefixes..\n");

    }


    for (int level = 0; level < levels; level++) {
        printf("Release Buffers for Level %i...\n", level);
        clReleaseMemObject(inputs[level]);
        clReleaseMemObject(prefix[level]);
    }

    return;


}