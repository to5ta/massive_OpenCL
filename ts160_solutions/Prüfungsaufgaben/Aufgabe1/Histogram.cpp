#include "Histogram.h"
#include <string.h>
#include "../shared/ansi_colors.h"
#include "../shared/clstatushelper.h"

OpenCLMgr *Histogram::OpenCLmgr = NULL;

Histogram::Histogram(){
    OpenCLmgr = new OpenCLMgr();
    OpenCLmgr->buildProgram("../Histogram.cl");
    const char *kernel_names[] = {"calcStatistic", "reduceStatistic"};
    OpenCLmgr->createKernels(kernel_names, 2);
}


Histogram::~Histogram(){
    delete [] hist;
    delete [] rgb_data;
}

void
Histogram::loadFile(char* filepath, int channels){
    int width, height, bpp;
    unsigned char* rgb;
    rgb = stbi_load( filepath, &width, &height, &bpp, 0 );
    // rgb is now three bytes per pixel, width*height size. Or NULL if load failed.
    // Do something with it...

    printf("Loading file:   %s\n", filepath);

    printf("Image bpp:    %8i\n", bpp);
    printf("Image Width:  %8i\n", width);
    printf("Image Height: %8i\n", height);

    this->height    = height;
    this->width     = width;

    if(rgb_data!=NULL){
        delete [] hist;
        delete [] rgb_data;
        hist= NULL;
        rgb_data=NULL;
    }

    hist        = new cl_uint[256]();
    rgb_data    = new unsigned char[width*height*3]();
    memcpy(rgb_data, rgb, width*height*3);

    stbi_image_free( rgb );
    return;
}


void
Histogram::plotData(){
    printf("\n\n");
    for(int y=0; y<height; y++){
        for(int x=0; x<width; x++){
            for (int i = 0; i < 3; ++i) {
                int val = rgb_data[x*3+y*3*width+i];
                if(val>255){
                    printf(ANSI_COLOR_RED);
                    printf("%3i", val);
                    printf(ANSI_COLOR_RESET);
                }
                else
                    printf("%3i", val);
                if(i<3)
                    printf("|");
            }
            printf("  ");
        }
        printf("\n");
    }
}


void
Histogram::calcHist(){
    cl_int status=0;

    int buffersize = width*height*3;

    printf("Buffersize: %i\n", buffersize);

    size_t gws[1] = {(width*height+8191)/8192*32 };
    size_t lws[1] = {32};
    int workgroups = gws[0]/lws[0];

    uint local_histograms[workgroups][256];



    cl_mem rgb_buffer = clCreateBuffer(OpenCLmgr->context,
                                       CL_MEM_READ_ONLY,
                                       buffersize*sizeof(u_char),
                                       NULL,
                                       NULL);

    cl_mem hist_buffer = clCreateBuffer(OpenCLmgr->context,
                                        CL_MEM_READ_ONLY,
                                        workgroups*256*sizeof(uint),
                                        NULL,
                                        NULL);

    status = clEnqueueWriteBuffer(OpenCLmgr->commandQueue,
                                  rgb_buffer,
                                  CL_TRUE,
                                  0,
                                  buffersize*sizeof(u_char),
                                  rgb_data,
                                  0,
                                  NULL,
                                  NULL);
    check_error(status);

    status = clSetKernelArg(OpenCLmgr->kernels["calcStatistic"],
                            0,
                            sizeof(cl_mem),
                            (void *) &rgb_buffer );
    check_error(status);

    status = clSetKernelArg(OpenCLmgr->kernels["calcStatistic"],
                            1,
                            sizeof(int),
                            (void *) &buffersize );
    check_error(status);

    status = clSetKernelArg(OpenCLmgr->kernels["calcStatistic"],
                            2,
                            sizeof(cl_mem),
                            (void *) &hist_buffer );
    check_error(status);

    printf("Global Work Size: %12i\n", gws[0]);
    printf("Local Work Size : %12i\n", lws[0]);
    printf("Work Groups     : %12i\n", workgroups);


    status = clEnqueueNDRangeKernel(OpenCLmgr->commandQueue,
                                    OpenCLmgr->kernels["calcStatistic"],
                                    1,
                                    NULL,
                                    gws,
                                    lws,
                                    0,
                                    NULL,
                                    NULL);
    check_error(status);

    status = clEnqueueReadBuffer( OpenCLmgr->commandQueue,
                                  hist_buffer,
                                  CL_TRUE,
                                  0,
                                  256*sizeof(uint),
                                  local_histograms,
                                  0,
                                  NULL,
                                  NULL);
    check_error(status);


//    for (int i = 0; i < workgroups; ++i) {
//        for (int j = 0; j < 256; ++j) {
//            printf("[%3i]: %12i; ",j, local_histograms[i][j]);
//            if((j+1)%10==0)
//                printf("\n");
//        }
//        printf("\n\n");
//    }
//

    clReleaseMemObject(rgb_buffer);


    plotData();


}