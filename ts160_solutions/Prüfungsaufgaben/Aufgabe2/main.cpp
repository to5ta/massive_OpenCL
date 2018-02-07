#include <iostream>
#include "../shared/clstatushelper.h"
#include "../shared/ansi_colors.h"
#include "PrefixSum.h"
#include <cmath>
#include <random>
#include <assert.h>

using namespace std;

int main() {


    PrefixSum prefixSum;

//    cl_uint numbers_to_sum[] = {1,4,2,12,34,1,6,7,9,8,9,5};
//    int dl = 12;

//    int dl = (int)(pow(256,2))+12345;
    int dl = (int) (pow(256, 3))*15 -30;
//    int dl = 256 *3-50;

//    cl_uint numbers_to_sum[dl];

    cl_uint *numbers_to_sum = (cl_uint *) (malloc(sizeof(cl_uint) * dl));
    assert(numbers_to_sum != NULL);


    for (cl_uint i = 0; i < dl; i++) {
        numbers_to_sum[i] = (int) (rand()%9)+1;
    }

    prefixSum.loadData(dl, numbers_to_sum);


    clock_t t;
    t = clock();
    prefixSum.prefixSumGPU();
    t = clock()-t;
    double gpu_dur = (double(t)) / CLOCKS_PER_SEC;

    t = clock();
    prefixSum.prefixSumCPU_Validate();
    t = clock()-t;
    double cpu_dur = (double(t)) / CLOCKS_PER_SEC;

    printf("GPU Duration: %5.3f ms\n", gpu_dur*1000.f);
    printf("CPU Duration: %5.3f ms\n", cpu_dur*1000.f);

    cout << "Aufgabe 2 [" << ANSI_COLOR_BRIGHTGREEN << "OK" << ANSI_COLOR_RESET << "]" << endl;
    return 0;
}