#include "PrefixSum.h"

OpenCLMgr* PrefixSum::OpenCLmgr = NULL;


PrefixSum::PrefixSum() {

    OpenCLmgr = new OpenCLMgr();
    OpenCLmgr->buildProgram("../prefixSum.cl");
    const char * kernel_names[] = {"prefixSum_kernel"};
    OpenCLmgr->createKernels(kernel_names, 1);

}

PrefixSum::~PrefixSum() {
    delete OpenCLmgr;
}

