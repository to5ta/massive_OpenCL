#include <iostream>
#include "../shared/ansi_colors.h"
#include "time.h"
#define STB_IMAGE_IMPLEMENTATION
#include "Histogram.h"
#include "unistd.h"

#include "string.h"

int main(int argc, char* argv[]) {

    clock_t t;
    Histogram histo(256, 32);

//    int i=0;
//    while(argv[i]!=NULL){
//        printf("\n %s is argv %d ",argv[i],i);
//        i++;
//    }
//
//    exit(0);

    if(argc>1){
        if( access( argv[1], F_OK ) != -1 ) {
            histo.loadFile(argv[1], 3);
        } else {
            std::cout << "[" << ANSI_COLOR_RED << "FAILED" << ANSI_COLOR_RESET  << "]";
            std::cout << argv[1] << " does not exist!" << std::endl;
        }
    }
    else {
        printf("No Image given! Use hardcoded path...\n");
//        histo.loadFile("../colorful_image_large.jpg", 3);
        histo.loadFile("../parrot_big.jpeg", 3);
    }


    t = clock();
    histo.calcHistGPU();
    t = clock() - t;
    double gpu_dur = (double(t)) / CLOCKS_PER_SEC;

    printf("\nGPU Histogram\n");
//    histo.plotHistogramTable(histo.hist);
    histo.plotHistogram(histo.hist);
//    histo.plotLocalHistograms(histo.local_histograms_gpu);


    t = clock();
    histo.calcHistCPU();
    t = clock() - t;
    double cpu_dur = (double(t)) / CLOCKS_PER_SEC;

    printf("\nCPU Histogram\n");
//    histo.plotHistogramTable(histo.hist_cpu);
    histo.plotHistogram(histo.hist_cpu);
//    histo.plotLocalHistograms(histo.local_histograms_cpu);

//    histo.compareGPUvsCPU();

//    histo.plotImageData(10);
//    histo.plotImageData(-10);


    if(memcmp(histo.hist_cpu, histo.hist, 256*sizeof(cl_uint))==0){
        std::cout << "[" << ANSI_COLOR_BRIGHTGREEN << "OK" << ANSI_COLOR_RESET  << "]";
        std::cout << " Result of GPU == Result of CPU" << std::endl;

        printf("GPU Duration: %5.3f ms\n", gpu_dur*1000.f);
        printf("CPU Duration: %5.3f ms\n", cpu_dur*1000.f);

    } else {
        std::cout << "[" << ANSI_COLOR_RED << "FAILED" << ANSI_COLOR_RESET  << "]";
        std::cout << " Result of GPU != Result of CPU" << std::endl;
    }
    return 0;
}