
#include <iostream>
#include "../shared/ansi_colors.h"
#include "time.h"
#define STB_IMAGE_IMPLEMENTATION
#include "Histogram.h"
#include "unistd.h"

#include "string.h"

int main(int argc, char* argv[]) {

    // args to parse
    char*   filename = NULL;
    int     out_of_order            = 0;
    int     pixels_per_workitem     = 256;
    int     groupsize               = 32;

    int options;

    while ((options = getopt (argc, argv, "of:p:g:")) != -1) {
        switch (options) {
            case 'o':
                out_of_order = 1;
                break;
            case 'f':
                filename = optarg;
                break;
            case 'p':
                pixels_per_workitem = atoi(optarg);
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

    printf("filename:            %s\n", filename);
    printf("out_of_order:        %i\n", out_of_order);
    printf("pixels_per_workitem: %i\n", pixels_per_workitem);
    printf("groupsize:           %i\n", groupsize);


    clock_t t;

    Histogram histo(pixels_per_workitem, groupsize, out_of_order);


    if(filename!=NULL){
        if( access( filename, F_OK ) != -1 ) {
            histo.loadFile(filename, 3);
        } else {
            std::cout << "[" << ANSI_COLOR_RED << "FAILED" << ANSI_COLOR_RESET  << "]";
            std::cout << filename << " does not exist!" << std::endl;
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