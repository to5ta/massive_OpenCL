#include "PrefixSum.h"
#include <cmath>
#include <cstring>
#include "../shared/clstatushelper.h"
#include "../shared/ansi_colors.h"
#include <assert.h>
#include "time.h"

#define SHORTEN_PLOT 1
#define SHOW_PLACEHOLDER 0
#define BLOCKSIZE_PLOT 1

OpenCLMgr *PrefixSum::OpenCLmgr = NULL;


PrefixSum::PrefixSum() {
    OpenCLmgr = new OpenCLMgr( CL_QUEUE_PROFILING_ENABLE );
    OpenCLmgr->loadFile("../prefixSum.cl");
    OpenCLmgr->buildProgram();
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
            printf("%5i", owndata[i]);
        else {
            printf(ANSI_COLOR_RED);
            printf("%5i", owndata[i]);
            printf(ANSI_COLOR_RESET);
        }

        if (i < length - 1)
            printf(", ");
        else
            printf("  ]");
        if ((i + 1) % columns == 0)
            printf("\n ");

        if ((i + 1) % blocksize == 0 && BLOCKSIZE_PLOT)
            printf("\n ");

        if (i + 1 == columns * lines && length > columns * lines * 2 + 1 && SHORTEN_PLOT) {
            printf("  ...\n ");
            i = length - 1 - columns * lines;
        }
    }
    printf("\n");
}


void PrefixSum::printInfo() {

    printf("Data Buffer Length: %5i\n", datalength);
    printf("Real Data Length:   %5i, %i dummy elements at the end.\n", reallength, datalength - reallength);

    if(SHOW_PLACEHOLDER){
        plotData(inputData, datalength);
    } else {
        plotData(inputData, reallength);
    }
}


void PrefixSum::plotCLBuffer(cl_mem buffer, int length) {
    cl_uint temp[length];
    cl_int status;
    status = clEnqueueReadBuffer(OpenCLmgr->commandQueue, buffer, CL_TRUE, 0, length * sizeof(int), temp, 0, NULL,
                                 NULL);
    check_error(status);

    plotData(temp, length);

}


void PrefixSum::addBlockSumToPrefix(cl_mem prefix0, cl_mem input1, int length0, int length1, int blocksize) {
    cl_int status = 0;

//    printf("Before AddSums:\n");
//    plotCLBuffer(prefix0, length0);

    status = clSetKernelArg(OpenCLmgr->kernels["addBlockSumToPrefix_kernel"], 0, sizeof(cl_mem), (void *) &prefix0);
    status |= clSetKernelArg(OpenCLmgr->kernels["addBlockSumToPrefix_kernel"], 1, sizeof(cl_mem), (void *) &input1);
    status |= clSetKernelArg(OpenCLmgr->kernels["addBlockSumToPrefix_kernel"], 2, sizeof(cl_int), (void *) &length0);
    status |= clSetKernelArg(OpenCLmgr->kernels["addBlockSumToPrefix_kernel"], 3, sizeof(cl_int), (void *) &length1);
    status |= clSetKernelArg(OpenCLmgr->kernels["addBlockSumToPrefix_kernel"], 4, sizeof(cl_int), (void *) &blocksize);
    check_error(status);

    size_t gws[1] = {blocksize};
    size_t lws[1] = {blocksize};

    status = clEnqueueNDRangeKernel(OpenCLmgr->commandQueue, OpenCLmgr->kernels["addBlockSumToPrefix_kernel"], 1, NULL, gws,
                                    lws, 0, NULL, NULL);
    check_error(status);
    return;
}


void PrefixSum::calcBlockSums(cl_mem input0, cl_mem prefix0, cl_mem input1, int length0, int length1, int blocksize) {
    cl_int status = 0;
    status = clSetKernelArg(OpenCLmgr->kernels["calcBlockSum_kernel"], 0, sizeof(cl_mem), (void *) &input0);
    status |= clSetKernelArg(OpenCLmgr->kernels["calcBlockSum_kernel"], 1, sizeof(cl_mem), (void *) &prefix0);
    status |= clSetKernelArg(OpenCLmgr->kernels["calcBlockSum_kernel"], 2, sizeof(cl_mem), (void *) &input1);
    status |= clSetKernelArg(OpenCLmgr->kernels["calcBlockSum_kernel"], 3, sizeof(cl_int), (void *) &length0);
    status |= clSetKernelArg(OpenCLmgr->kernels["calcBlockSum_kernel"], 4, sizeof(cl_int), (void *) &length1);
    status |= clSetKernelArg(OpenCLmgr->kernels["calcBlockSum_kernel"], 5, sizeof(cl_int), (void *) &blocksize);
    check_error(status);

    size_t gws[1] = {blocksize};
    size_t lws[1] = {blocksize};

    status = clEnqueueNDRangeKernel(OpenCLmgr->commandQueue, OpenCLmgr->kernels["calcBlockSum_kernel"], 1, NULL, gws,
                                    lws, 0, NULL, NULL);
    check_error(status);

    return;
}


void PrefixSum::prefixBlockwise(cl_mem input0, cl_mem prefix0, int length0, int blocksize) {

    cl_int status = 0;
    status = clSetKernelArg(OpenCLmgr->kernels["prefixBlockwise_kernel"], 0, sizeof(cl_mem), (void *) &input0);
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

    cl_int status;

    int levels = (int) (ceil((log(datalength) / log(256))));

    cl_mem inputs[levels];
    cl_mem prefix[levels];
    int bufferLengths[levels];
    int dataLengths[levels];

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

    inputs[0] = clCreateBuffer(OpenCLmgr->context,
                               CL_MEM_READ_ONLY,
                               datalength * sizeof(cl_uint),
                               NULL,
                               NULL);

    status = clEnqueueWriteBuffer(OpenCLmgr->commandQueue,
                                  inputs[0],
                                  CL_TRUE,
                                  0,
                                  datalength * sizeof(cl_uint),
                                  inputData,
                                  0,
                                  NULL,
                                  NULL);

    check_error(status);

    prefix[0] = clCreateBuffer(OpenCLmgr->context,
                               CL_MEM_READ_WRITE,
                               datalength * sizeof(cl_uint),
                               NULL,
                               NULL);

    // buffer Length Size
    int b_len = datalength;
    int d_len = reallength;
    bufferLengths[0] = b_len;
    dataLengths[0] = d_len;

    // create level buffers
    for (int level = 1; level < levels; level++) {

        d_len = ((b_len - 1) / 256 + 1);
        b_len = ((d_len - 1) / 256 + 1) * 256;

        bufferLengths[level] = b_len;
        dataLengths[level] = d_len;

        printf("Creating Buffers of Length %5i for Level %i...\n", b_len, level);

        inputs[level] = clCreateBuffer(OpenCLmgr->context, CL_MEM_READ_WRITE, b_len * sizeof(cl_uint), NULL, NULL);
        prefix[level] = clCreateBuffer(OpenCLmgr->context, CL_MEM_READ_WRITE, b_len * sizeof(cl_uint), NULL, NULL);
    }

    printf("\n----------------\n\n");

    // do the blockwise prefix stuff repetitive
    for (int level = 0; level < levels; level++) {

        if (level > 0) {
            // blocksum aka new input data
            printf("Calculate Block Sums for Level %i (each two last of a block)...\n", level);
            calcBlockSums(inputs[level - 1], prefix[level - 1], inputs[level], dataLengths[level - 1],
                          dataLengths[level], blocksize);
//            plotCLBuffer(inputs[level], bufferLengths[level]);
        }

        // simple blockwise prefixes
        printf("Calculate Prefix Blockwise for Level %i...\n", level);
        prefixBlockwise(inputs[level], prefix[level], bufferLengths[level], blocksize);
        if (level == 1) {
//            printf("Blockwise Prefix Sums Level 1...\n");
//            plotCLBuffer(prefix[1], bufferLengths[1]);
        }
    }

    printf("\n----------------\n\n");

    // build real prefixes, add previous prefix sums (block by block internal + block sum)
    for (int output = levels - 1; output > 0; output--) {
        // addBlockSumToPrefix
        printf("Add Prefixed Block Sums L%i to Blockwise Prefixes L%i...\n", output, output - 1);
//        plotCLBuffer(inputs[output], bufferLengths[output]);
        addBlockSumToPrefix(prefix[output-1], prefix[output], dataLengths[output-1], dataLengths[output], blocksize);
    }

    status = clEnqueueReadBuffer(OpenCLmgr->commandQueue, prefix[0], CL_TRUE, 0, datalength * sizeof(int), prefixData,
                                 0, NULL, NULL);
    check_error(status);

    printf("Result: \n");

    if(SHOW_PLACEHOLDER){
        plotData(prefixData, datalength);
    } else {
        plotData(prefixData, reallength);
    }

    for (int level = 0; level < levels; level++) {
        printf("Release CL_Buffers for Level %i...\n", level);
        clReleaseMemObject(inputs[level]);
        clReleaseMemObject(prefix[level]);
    }

    return;

}



void
PrefixSum::prefixSumCPU_Validate() {
    cl_uint* cpuPrefixSums = new cl_uint[reallength]();

    cl_uint prefix = 0;

    for (int i = 1; i < reallength; ++i) {
        cpuPrefixSums[i] = cpuPrefixSums[i-1]+inputData[i-1];
    }

    int err = 0;
    for (int i = 0; i < reallength; ++i) {
        if(this->prefixData[i]!=cpuPrefixSums[i]){
            printf(ANSI_COLOR_RED);
            printf(ANSI_BOLD);
            printf("CPU != GPU\n");
            printf("i: %i, GPU: %i, CPU: %i\n",i, prefixData[i], cpuPrefixSums[i]);
            printf(ANSI_COLOR_RESET);
            err++;
            break;
        }
    }

    if(!err){
        printf(ANSI_COLOR_GREEN);
        printf(ANSI_BOLD);
        printf("CPU == GPU\n");
        printf(ANSI_COLOR_RESET);
    }

    plotData(cpuPrefixSums, reallength);

    delete [] cpuPrefixSums;
}