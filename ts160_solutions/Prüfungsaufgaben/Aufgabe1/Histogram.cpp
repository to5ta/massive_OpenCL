#include "Histogram.h"
#include <string.h>
#include <vector>
#include "../shared/ansi_colors.h"
#include "../shared/clstatushelper.h"

OpenCLMgr *Histogram::OpenCLmgr = NULL;

Histogram::Histogram(){
    OpenCLmgr = new OpenCLMgr();
    OpenCLmgr->buildProgram("../Histogram.cl");
    const char *kernel_names[] = {"calcStatistic_kernel", "reduceStatistic_kernel"};
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

    printf("Image bpp:    %10i\n", bpp);
    printf("Image Width:  %10i\n", width);
    printf("Image Height: %10i\n", height);

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
Histogram::plotHistogram(){
    if(hist==NULL){
        printf("hist == NULL!");
        return;
    }


    for (int j = 0; j < 256; ++j) {
        printf("[%3i]: %12i; ",j, hist[j]);

        if((j+1)%10==0)
            printf("\n");
    }

    printf("\n\n");


    printf("      ");
    for (int i = 0; i < 255; ++i) {
        printf("-");
    }
    printf("\n");

    for (int p = 20; p >= 0; --p) {
        printf("%3i% |", p*5);
        for (int b = 0; b < 256; ++b) {
            float c = (float)(hist[b]);

//            if(b==255)
//                printf("%f\n", c);

            float s = (float)(width*height);
            float percent = c/s *100.f;
            if(percent>(p*5)){
                printf("#");
            }
            else{
                printf(" ");
            }
        }
        printf("\n");
    }

    printf("      ");
    for (int k = 0; k < 16; ++k) {
        printf("%-16c", '|');
    }
    printf("\n      ");
    for (int k = 0; k < 16; ++k) {
        printf("%-16i", k*16);
    }
    printf("\n");
    printf("\n\n");

}




void
Histogram::plotImageData(){
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

    printf("Buffersize: %12i\n", buffersize);

    size_t gws[1] = {(width*height+8191)/8192*32 };
    size_t lws[1] = {32};
    int workgroups = gws[0]/lws[0];

    uint * local_histograms = new uint[workgroups*256]();


    cl_mem rgb_buffer = clCreateBuffer(OpenCLmgr->context,
                                       CL_MEM_READ_ONLY,
                                       buffersize*sizeof(u_char),
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



    cl_mem all_hist_buffer = clCreateBuffer(OpenCLmgr->context,
                                        CL_MEM_READ_WRITE,
                                        workgroups*256*sizeof(uint),
                                        NULL,
                                        NULL);

    cl_mem hist_buffer = clCreateBuffer(OpenCLmgr->context,
                                        CL_MEM_READ_WRITE,
                                        256*sizeof(uint),
                                        NULL,
                                        NULL);

    status = clEnqueueWriteBuffer(OpenCLmgr->commandQueue,
                                  all_hist_buffer,
                                  CL_TRUE,
                                  0,
                                  workgroups*256*sizeof(uint),
                                  local_histograms,
                                  0,
                                  NULL,
                                  NULL);
    check_error(status);



    status = clSetKernelArg(OpenCLmgr->kernels["calcStatistic_kernel"],
                            0,
                            sizeof(cl_mem),
                            (void *) &rgb_buffer );
    check_error(status);

    status = clSetKernelArg(OpenCLmgr->kernels["calcStatistic_kernel"],
                            1,
                            sizeof(int),
                            (void *) &buffersize );
    check_error(status);

    status = clSetKernelArg(OpenCLmgr->kernels["calcStatistic_kernel"],
                            2,
                            sizeof(cl_mem),
                            (void *) &all_hist_buffer );
    check_error(status);

    printf("Global Work Size: %12i\n", gws[0]);
    printf("Local Work Size : %12i\n", lws[0]);
    printf("Work Groups     : %12i\n", workgroups);


    status = clEnqueueNDRangeKernel(OpenCLmgr->commandQueue,
                                    OpenCLmgr->kernels["calcStatistic_kernel"],
                                    1,
                                    NULL,
                                    gws,
                                    lws,
                                    0,
                                    NULL,
                                    NULL);
    check_error(status);

    status = clEnqueueReadBuffer( OpenCLmgr->commandQueue,
                                  all_hist_buffer,
                                  CL_TRUE,
                                  0,
                                  workgroups*256*sizeof(uint),
                                  local_histograms,
                                  0,
                                  NULL,
                                  NULL);
    check_error(status);

    // reduce
    status = clSetKernelArg(OpenCLmgr->kernels["reduceStatistic_kernel"],
                            0,
                            sizeof(cl_mem),
                            (void *) &all_hist_buffer );
    check_error(status);

    status = clSetKernelArg(OpenCLmgr->kernels["reduceStatistic_kernel"],
                            1,
                            sizeof(int),
                            (void *) &workgroups );
    check_error(status);

    status = clSetKernelArg(OpenCLmgr->kernels["reduceStatistic_kernel"],
                            2,
                            sizeof(cl_mem),
                            (void *) &hist_buffer );
    check_error(status);

    status = clEnqueueNDRangeKernel(OpenCLmgr->commandQueue,
                                    OpenCLmgr->kernels["reduceStatistic_kernel"],
                                    1,
                                    NULL,
                                    lws,
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
                                  this->hist,
                                  0,
                                  NULL,
                                  NULL);
    check_error(status);




//    memcpy(hist, local_histograms, 256*sizeof(uint));
//

//    for (int i = 0; i < workgroups; ++i) {
//        printf("LOCAL HISTOGRAM: %i\n", i);
//        for (int j = 0; j < 256; ++j) {
//            printf("[%3i]: %12i; ",j, local_histograms[i*256+j]);
//
//            if((j+1)%10==0)
//                printf("\n");
//        }
//        printf("\n\n");
//    }

    clReleaseMemObject(rgb_buffer);
    clReleaseMemObject(hist_buffer);
    clReleaseMemObject(all_hist_buffer);

    delete [] local_histograms;




    plotImageData();
    plotHistogram();


}