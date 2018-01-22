//
// Created by tosta on 22.01.18.
//

#ifndef AUFGABE1_HISTOGRAM_H
#define AUFGABE1_HISTOGRAM_H

#include "../shared/OpenCLMgr.h"

#include "../shared/stb_image.h"

class Histogram {

public:

    Histogram();
    ~Histogram();

    void loadFile(char* filepath, int channels);
    void calcHist(char* rgb_data);

    static OpenCLMgr * OpenCLmgr;

    cl_uint *data = NULL;
    cl_uint *hist = NULL;
    int datalength = 0;
//    int reallength = 0;



};


#endif //AUFGABE1_HISTOGRAM_H
