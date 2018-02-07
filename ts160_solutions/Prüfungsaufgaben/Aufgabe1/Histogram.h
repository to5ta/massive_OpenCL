#ifndef AUFGABE1_HISTOGRAM_H
#define AUFGABE1_HISTOGRAM_H

#include "../shared/OpenCLMgr.h"
#include "../shared/stb_image.h"

class Histogram {

public:

    Histogram(int pixels_per_workitem,
              int group_size,
              int atomic_add,
              int out_of_order);

    ~Histogram();

    static OpenCLMgr * OpenCLmgr;

    void loadFile(char* filepath, int channels);
    void calcHistGPU();
    void calcHistGPUwithEvents();
    void calcHistCPU();

    void plotImageData(int max_id);
    void plotHistogram(cl_uint * histo);
    void plotHistogramTable(cl_uint * histo);
    void plotLocalHistograms(cl_uint * local_histo);

    void compareGPUvsCPU();

    unsigned char *rgb_data = NULL;
    cl_uint *hist = NULL;
    cl_uint *local_histograms_gpu = NULL;
    cl_uint *hist_cpu = NULL;
    cl_uint *local_histograms_cpu = NULL;
    int datalength = 0;

private:
    int height = 0;
    int width = 0;
    int bytes_per_pixel = 0;

    int buffersize;
    int pixels_per_workitem;
    int group_size;
    int pixels_per_group;
    int workgroups;

    size_t gws[1] = {0};
    size_t lws[1] = {0};

};


#endif //AUFGABE1_HISTOGRAM_H
