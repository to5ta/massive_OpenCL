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

//    cl_uint data[] = {1,2,3,43,4,7,56,9,5,6,22,12,32,45,77,0};
//    cl_uint sort[] = {0,1,2,3,4,5,6,7,9,12,22,32,43,45,56,77};
//    int dl = 16;

//    cl_uint data[] = {1,2,3,43,4,7,56,9,5,6,22,12,32,45,77,0, 555};
//    cl_uint sort[] = {0,1,2,3,4,5,6,7,9,12,22,32,43,45,56,77, 555};
//    int dl = 17;

//
//    cl_uint data[] = {268, 156, 148, 602, 344, 365, 746, 414, 137, 779, 864, 929, 162,  88, 182,  94, 597, 193, 588, 913,
//                      805, 557, 722, 228, 515, 903,  50, 239, 675, 503, 924, 313,  47, 923, 925, 160, 717, 105, 644, 583,
//                      937, 748, 178, 980, 780, 978,  99, 767,  91, 660, 774, 167,  33, 679, 327, 260, 356, 436, 156, 732};
//
//    cl_uint sort[] = { 33,  47,  50,  88,  91,  94,  99, 105, 137, 148, 156, 156, 160, 162, 167, 178, 182, 193, 228, 239,
//                      260, 268, 313, 327, 344, 356, 365, 414, 436, 503, 515, 557, 583, 588, 597, 602, 644, 660, 675, 679,
//                      717, 722, 732, 746, 748, 767, 774, 779, 780, 805, 864, 903, 913, 923, 924, 925, 929, 937, 978, 980};
//    int dl = 60;



    uint dl = 512;


    cl_uint *numbers_to_sort = (cl_uint *) (malloc(sizeof(cl_uint) * dl));
    assert(numbers_to_sort != nullptr);

    for (cl_uint i = 0; i < dl; i++) {
        numbers_to_sort[i] = (int) (rand()%9999)+1;
    }


    printf("INPUT: 0x%x\n", numbers_to_sort);
    bitonicSort->printData(numbers_to_sort, dl, 10);

    bitonicSort->loadData(dl, numbers_to_sort);

    bitonicSort->sortGPU();
//    bitonicSort->sortGPU2();

    //host-side loops
    // bitonicSort->sortGPU2();

    printf("GPU: 0x%x\n", bitonicSort->gpu_data);
    bitonicSort->printData(bitonicSort->gpu_data, dl, 10);

    bitonicSort->sortCPU();
    printf("CPU: 0x%x\n", bitonicSort->cpu_data);
    bitonicSort->printData(bitonicSort->cpu_data, dl, 10);

//    for(int i=0; i<dl; i++){
//        cout << bitonicSort->data[i] << endl;
//    }





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

    return 0;
}