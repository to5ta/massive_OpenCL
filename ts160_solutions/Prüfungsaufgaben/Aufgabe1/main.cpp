#include <iostream>
#include "../shared/ansi_colors.h"
#include "time.h"


#define STB_IMAGE_IMPLEMENTATION

#include "Histogram.h"



int main() {

    Histogram histo(64, 32);

//    histo.loadFile("colorful_image_small.jpg", 3);
//    histo.loadFile("../colorful_image_small.jpg", 3);
    histo.loadFile("../colorful_image_large.jpg", 3);
//    histo.loadFile("../color  _test_mini.jpg", 3);
//    histo.loadFile("../color_test_mini.png", 3);

//    histo.loadFile("../gradient_bw.png", 3);
//    histo.loadFile("../ease_bw.png", 3);
//    histo.loadFile("../half_ease_large.png", 3);
//    histo.loadFile("../gradient_small.png", 3);

//    histo.loadFile("../stripes_bw.png", 3);
//    histo.loadFile("../hist2parts.png", 3);
//    histo.loadFile("../parrot.jpeg", 3);
//    histo.loadFile("../parrot_big.jpeg", 3);
//    histo.loadFile("../parrot_medium.jpeg", 3);
//    histo.loadFile("../parrot_medium2.jpeg", 3);


    long gpu_start = clock();
    histo.calcHistGPU();
    long gpu_end = clock();
    printf("\nGPU Histogram\n");
//    histo.plotHistogramTable(histo.hist);
    histo.plotHistogram(histo.hist);
//    histo.plotLocalHistograms(histo.local_histograms_gpu);


    long cpu_start = clock();
    histo.calcHistCPU();
    long cpu_end = clock();
    printf("\nCPU Histogram\n");
//    histo.plotHistogramTable(histo.hist_cpu);
    histo.plotHistogram(histo.hist_cpu);
//    histo.plotLocalHistograms(histo.local_histograms_cpu);


    histo.compareGPUvsCPU();

//    histo.plotImageData(10);
//    histo.plotImageData(-10);

//
//    if(histo.plotHistogram(0)==0){
//        std::cout << "[" << ANSI_COLOR_BRIGHTGREEN << "OK" << ANSI_COLOR_RESET  << "]";
//        std::cout << " Sum of Bins == Sum of Pixels" << std::endl;
//    } else {
//        std::cout << "[" << ANSI_COLOR_RED << "FAILED" << ANSI_COLOR_RESET  << "]";
//        std::cout << " Sum of Bins != Sum of Pixels" << std::endl;
//    }
//

    if(memcmp(histo.hist_cpu, histo.hist, 256*sizeof(cl_uint))==0){
        std::cout << "[" << ANSI_COLOR_BRIGHTGREEN << "OK" << ANSI_COLOR_RESET  << "]";
        std::cout << " Result of GPU == Result of CPU" << std::endl;

        float gpu_dur = float(gpu_end-gpu_start)/1000.f;
        float cpu_dur = float(cpu_end-cpu_start)/1000.f;
        std::cout << "GPU Duration: " << gpu_dur << " s" << std::endl;
        std::cout << "CPU Duration: " << cpu_dur << " s" << std::endl;

    } else {
        std::cout << "[" << ANSI_COLOR_RED << "FAILED" << ANSI_COLOR_RESET  << "]";
        std::cout << " Result of GPU != Result of CPU" << std::endl;
    }
    return 0;
}