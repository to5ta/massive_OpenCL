#ifndef AUFGABE2_PREFIXSUM_H
#define AUFGABE2_PREFIXSUM_H

#include "../shared/OpenCLMgr.h"

class PrefixSum {

public:
    PrefixSum();
    virtual ~PrefixSum();

    static OpenCLMgr * OpenCLmgr;


    void loadData(int newdatalength, cl_uint *newdata);
    void prefixSumGPU();
    void printInfo();

    cl_uint *inputData = NULL;
    cl_uint *prefixData = NULL;

    int datalength = 0;
    int reallength = 0;

private:
    void plotData(cl_uint* owndata);

};







#endif //AUFGABE2_PREFIXSUM_H
