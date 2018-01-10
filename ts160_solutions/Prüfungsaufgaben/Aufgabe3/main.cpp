#include <iostream>
#include "../shared/clstatushelper.h"
#include "../shared/ansi_colors.h"
#include "BitonicSort.h"
#include <cstring>

using namespace std;

int main(int argc, char* argv[]) {



    BitonicSort * bitonicSort = new BitonicSort();
//    OpenCLMgr oclmgr;
//    bitonicSort ->OpenCLmgr = &oclmgr;

    cl_uint data[] = {1,2,3,43,4,7,56,9,5,6,22,12,32,45,77,0};
    int dl = 16;

    printf("-");
    for(int i=0; i<dl; i++){
        printf("%2i", i);
        if(i<dl-1) {
            printf(", ");
        }
    }
    printf("-\n");

    printf("[");
    for(int i=0; i<dl; i++){
        printf("%2i", data[i]);
        if(i<dl-1) {
            printf(", ");
        }
    }
    printf("]\n");



    bitonicSort->loadData(dl, data);
    bitonicSort->sortGPU();

    for(int i=0; i<dl; i++){
        cout << bitonicSort->data[i] << endl;
    }



    cout  << "Aufgabe 3 ["<<  ANSI_COLOR_BRIGHTGREEN << "OK" << ANSI_COLOR_RESET << "]" << endl;
    return 0;
}