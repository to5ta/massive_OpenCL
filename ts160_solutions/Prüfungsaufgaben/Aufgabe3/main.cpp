#include <iostream>
#include "../shared/clstatushelper.h"
#include "../shared/ansi_colors.h"
#include "BitonicSort.h"
#include <cstring>
#include <assert.h>
#include <cmath>


// host-side loops
#define HOST_SIDE_LOOPS 0


using namespace std;


int main(int argc, char* argv[]) {


    BitonicSort * bitonicSort = new BitonicSort();

//    uint dl = 16000;
    uint dl = 65536;
//    uint dl = 131000;

    cl_uint *numbers_to_sort = (cl_uint *) (malloc(sizeof(cl_uint) * dl));
    assert(numbers_to_sort != nullptr);

    for (cl_uint i = 0; i < dl; i++) {
        numbers_to_sort[i] = (int) (rand()%9999)+1;
    }



    printf("INPUT: 0x%x\n", numbers_to_sort);
    bitonicSort->printData(numbers_to_sort, dl, 10);

    bitonicSort->loadData(dl, numbers_to_sort);

//    bitonicSort->sortGPU();

    //host-side loops
     bitonicSort->sortGPU2();

    printf("GPU: 0x%x\n", bitonicSort->gpu_data);
    bitonicSort->printData(bitonicSort->gpu_data, dl, 10);

    bitonicSort->sortCPU();
    printf("CPU: 0x%x\n", bitonicSort->cpu_data);
    bitonicSort->printData(bitonicSort->cpu_data, dl, 10);


    if(memcmp(bitonicSort->gpu_data, bitonicSort->cpu_data, dl)==0) {
        cout  << "GPU == CPU ["<<  ANSI_COLOR_BRIGHTGREEN << "OK" << ANSI_COLOR_RESET << "]" << endl;
    } else {

        int errors = 0;

        // manual check, again
        for (int i = 0; i < dl; ++i) {
            if( bitonicSort->gpu_data[i]!=bitonicSort->cpu_data[i]){
//                printf("%i: GPU: %i, CPU: %i\n", i, bitonicSort->gpu_data[i], bitonicSort->cpu_data[i]);
                errors++;
            }
        }

        if(!errors){
            cout  << "GPU == CPU ["<<  ANSI_COLOR_YELLOW << "OK" << ANSI_COLOR_RESET << "], but memcmp!=0" << endl;
        } else {
            cout  << "GPU != CPU, "<< errors << " Abweichungen ["<<  ANSI_COLOR_RED << "FAILED" << ANSI_COLOR_RESET << "]" << endl;

        }
    }

    delete bitonicSort;

    return 0;
}