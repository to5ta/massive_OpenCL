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

    cl_uint data[] = {1,2,3,43,4,7,56,9,5,6};

    bitonicSort->loadData(10, data);
    bitonicSort->sortGPU();

    for(int i=0; i<10; i++){
        cout << bitonicSort->data[i] << endl;
    }



    cout  << "Aufgabe 3 ["<<  ANSI_COLOR_BRIGHTGREEN << "OK" << ANSI_COLOR_RESET << "]" << endl;
    return 0;
}