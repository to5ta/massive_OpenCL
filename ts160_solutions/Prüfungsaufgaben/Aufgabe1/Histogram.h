#ifndef AUFGABE1_HISTOGRAM_H
#define AUFGABE1_HISTOGRAM_H

#include "../shared/OpenCLMgr.h"
#include "../shared/stb_image.h"

class Histogram {

public:

    Histogram();
    ~Histogram();

    static OpenCLMgr * OpenCLmgr;

    void loadFile(char* filepath, int channels);
    void calcHistGPU();
    void calcHistCPU();

    void plotImageData();
    void plotHistogram(cl_uint * histo);
    void plotHistogramTable(cl_uint * histo);


    unsigned char *rgb_data = NULL;
    cl_uint *hist = NULL;
    cl_uint *hist_cpu = NULL;
    int datalength = 0;
//    int reallength = 0;

private:
    int height = 0;
    int width = 0;
    int bpp = 0;

};


#endif //AUFGABE1_HISTOGRAM_H
