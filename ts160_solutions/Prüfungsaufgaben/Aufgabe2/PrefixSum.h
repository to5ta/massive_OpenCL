#ifndef AUFGABE2_PREFIXSUM_H
#define AUFGABE2_PREFIXSUM_H

#include "../shared/OpenCLMgr.h"

class PrefixSum {

public:
    PrefixSum();

    virtual ~PrefixSum();

    static OpenCLMgr *OpenCLmgr;


    void loadData(int newdatalength, cl_uint *newdata);

    void prefixSumGPU();

    void printInfo();

    cl_uint *inputData = NULL;
    cl_uint *prefixData = NULL;

    int datalength = 0;
    int reallength = 0;

    int blocksize = 256;

private:
    void plotData(cl_uint *owndata, int length);
    void plotCLBuffer(cl_mem, int length);


    // void calculateLevelBlocks(cl_mem input0, cl_mem prefix0, cl_mem input1, cl_mem prefix1, int length0,int length1);

    void addBlockSumToPrefix(   cl_mem  prefix0,
                                cl_mem  input1,
                                int     length0,
                                int     length1,
                                int     blocksize);

    void calcBlockSums(     cl_mem  input0,
                            cl_mem  prefix0,
                            cl_mem  input1,
                            int     length0,
                            int     length1,
                            int     blocksize);

    void prefixBlockwise(   cl_mem  input0,
                            cl_mem  prefix0,
                            int     length0,
                            int     blocksize);
};

#endif //AUFGABE2_PREFIXSUM_H
