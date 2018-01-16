#ifndef AUFGABE3_BITONICSORT_H
#define AUFGABE3_BITONICSORT_H

#include "../shared/OpenCLMgr.h"

class BitonicSort {

public:
    BitonicSort();

    ~BitonicSort();

    void loadData(int datalength, cl_uint *newdata);
    void sortGPU();
    void sortCPU();

    static OpenCLMgr * OpenCLmgr;

    cl_uint *data = NULL;
    int datalength = 0;
    int reallength = 0;

private:


};


#endif //AUFGABE3_BITONICSORT_H
