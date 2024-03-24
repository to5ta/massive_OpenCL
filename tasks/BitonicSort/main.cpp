#include <iostream>
#include "../shared/clstatushelper.h"
#include "../shared/ansi_colors.h"
#include "BitonicSort.h"
#include <cstring>
#include <assert.h>
#include "unistd.h"
#include <cmath>


using namespace std;


int main(int argc, char* argv[]) {


    int     host_side_loops = 0;
    uint    dl             = 16000;

    clock_t t;

    int options;

    while ((options = getopt (argc, argv, "hl:")) != -1) {
        switch (options) {
            case 'h':
                host_side_loops = 1;
                printf("\n-h: Calculating with host-side-loops, multiple starts of the kernel!\n\n");
                break;

            case 'l':
                dl = atoi(optarg);
                printf("\n-l <n>: Generating pseudo-random input data of length %i!\n\n", dl);
                break;

            case '?':
                if (optopt == 'c')
                    fprintf(stderr, "Option -%c requires an argument.\n", optopt);
                else if (isprint(optopt))
                    fprintf(stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf(stderr,
                            "Unknown option character `\\x%x'.\n",
                            optopt);
                return 1;

            default:
                abort();
        }
    }


    BitonicSort * bitonicSort = new BitonicSort(host_side_loops);

    cl_uint *numbers_to_sort = (cl_uint *) (malloc(sizeof(cl_uint) * dl));
    assert(numbers_to_sort != nullptr);

    for (cl_uint i = 0; i < dl; i++) {
        numbers_to_sort[i] = (int) (rand()%9999)+1;
    }

    printf("INPUT: 0x%x\n", numbers_to_sort);
    bitonicSort->printData(numbers_to_sort, dl, 10);

    bitonicSort->loadData(dl, numbers_to_sort);

    t = clock();
    if(host_side_loops){
        //host-side loops
        bitonicSort->sortGPU2();

    } else {
        bitonicSort->sortGPU();
    }
    t = clock() - t;
    double gpu_dur = (double(t)) / CLOCKS_PER_SEC;


    printf("GPU: 0x%x\n", bitonicSort->gpu_data);
    bitonicSort->printData(bitonicSort->gpu_data, dl, 10);

    t = clock();
    bitonicSort->sortCPU();
    printf("CPU: 0x%x\n", bitonicSort->cpu_data);
    bitonicSort->printData(bitonicSort->cpu_data, dl, 10);
    t = clock() - t;
    double cpu_dur = (double(t)) / CLOCKS_PER_SEC;

    printf("GPU Duration: %5.3f ms\n", gpu_dur*1000.f);
    printf("CPU Duration: %5.3f ms\n", cpu_dur*1000.f);

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