#include "Histogram.h"
#include <cstring>
#include <vector>
#include <assert.h>
#include <cmath>
#include <unistd.h>
#include "../shared/ansi_colors.h"
#include "../shared/clstatushelper.h"



#define PRINT_LOCAL_HISTOGRAMS 1

OpenCLMgr *Histogram::OpenCLmgr = nullptr;

Histogram::Histogram(int pixel_per_workitem,
                     int group_size){

    this->pixels_per_workitem = pixel_per_workitem;
    this->group_size          = group_size;

    OpenCLmgr = new OpenCLMgr();
    OpenCLmgr->buildProgram("../Histogram3.cl");
    const char *kernel_names[] = {"calcStatistic_kernel", "reduceStatistic_kernel"};
    OpenCLmgr->createKernels(kernel_names, 2);
}


Histogram::~Histogram(){
    delete [] hist;
    delete [] rgb_data;
//    free(local_histograms_gpu);
//    free(local_histograms_cpu);

    delete [] local_histograms_gpu;
    delete [] local_histograms_cpu;
}


void
Histogram::loadFile(char* file_path, int channels){
    printf("Loading file:   %s\n", file_path);

    int w, h, bpp;
    unsigned char* rgb;
    rgb = stbi_load( file_path, &w, &h, &bpp, 0 );

    // rgb is now three bytes per pixel, width*height size. Or nullptr if load failed.

    printf("%-25s: %12i\n", "Bytes per Pixel", bpp);
    printf("%-25s: %12i\n", "Width", w);
    printf("%-25s: %12i\n", "Height", h);
    printf("%-25s: %12i\n", "Pixels", w*h);
    printf("%-25s: %12i\n", "Bytes per Pixel", bpp);

    this->height            = h;
    this->width             = w;
    this->bytes_per_pixel   = bpp;

    if(bpp!=channels){
        printf("ERROR: bpp and channels count does not match!");
    }

    if(rgb_data!=nullptr) {
        delete[] rgb_data;
        rgb_data = nullptr;
    }
    if(hist!=nullptr){
        delete [] hist;
        hist= nullptr;
    }

    if(hist_cpu!=nullptr){
        delete [] hist_cpu;
        hist_cpu= nullptr;
    }

    hist        = new cl_uint[256]();
    assert(hist!= nullptr);
    hist_cpu    = new cl_uint[256]();
    assert(hist_cpu!= nullptr);
    rgb_data    = new unsigned char[width*height*channels]();
    memcpy(rgb_data, rgb, width*height*channels);
    stbi_image_free( rgb );


    buffersize = width*height*channels;
    datalength = buffersize;
    printf("%-25s: %12i\n", "Buffersize (bytes)", buffersize);


    pixels_per_group    = pixels_per_workitem * group_size;
    printf("\n%-25s: %12i\n", "Pixels per Workitem", pixels_per_workitem);
    printf("%-25s: %12i\n", "Pixels per Group", pixels_per_group);
    printf("%-25s: %12i\n", "Group Size", group_size);

    int _gws = ((width*height)+pixels_per_group-1)/pixels_per_group * group_size;
    gws[0] = size_t(_gws);
    lws[0] = size_t(group_size);
    workgroups = gws[0]/lws[0];

    printf("\n%-25s: %12i\n", "GLOBAL WORK SIZE",int(gws[0]));
    printf("%-25s: %12i\n", "LOCAL WORK SIZE",int(lws[0]));
    printf("%-25s: %12i\n", "WORK GROUPS", workgroups);
    printf("%-25s: %12i\n", "GWS * Pixels per Group", workgroups*pixels_per_group);

    // CL_DEVICE_MAX_WORK_GROUP_SIZE
    if(workgroups>1024){
        printf(ANSI_BOLD);
        printf(ANSI_COLOR_RED);
        printf("ERROR: GROUP SIZE < 1024 (HARDWARE MAX)!");
        printf(ANSI_COLOR_RESET);
        exit(0);
    }

//    local_histograms_gpu = (cl_uint*)(malloc(workgroups*256*sizeof(cl_uint)));
//    local_histograms_cpu = (cl_uint*)(malloc(workgroups*256*sizeof(cl_uint)));

    local_histograms_gpu = new cl_uint[workgroups*256]();
    local_histograms_cpu = new cl_uint[workgroups*256]();

    assert(local_histograms_gpu!= nullptr);
    assert(local_histograms_cpu!= nullptr);

    return;
}




void
Histogram::plotHistogramTable(cl_uint * histo) {

    for (int j = 0; j < 256; ++j) {
        printf("[%3i]: %12i; ", j, histo[j]);

        if ((j + 1) % 10 == 0) {
            printf("\n");
        }
    }
    printf("\n\n");

}



void
Histogram::plotHistogram(cl_uint * histogram){
    printf(ANSI_BOLD);
    if(histogram== nullptr){
        printf("histogram == nullptr!");
        return;
    }
    // find max
    int max_bin = 0;
    for (int l = 0; l < 256; ++l) {
        if(histogram[l]>max_bin){
            max_bin = histogram[l];
        }
    }
    cl_uint total_count = 0;
    printf("      ");
    for (int i = 0; i < 255; ++i) {
        printf("-");
    }
    printf("\n");
    for (int p = 20; p >= 0; --p) {
        printf("%3i% |", p*5);
        for (int b = 0; b < 256; ++b) {
            float c = (float)(histogram[b]);
            if(p==0){
                total_count += histogram[b];
            }
            float s = (float)(max_bin);
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
    for (int k = 0; k < 52; ++k) {
        printf("%-5c", '^');
    }
    printf("\n      ");
    for (int k = 0; k < 52; ++k) {
        printf("%-5i", k*5);
    }
    printf("\n\n");
    printf("Sum Bins : %12i\n", total_count);
    printf("Pixels   : %12i\n", height*width);
    printf(ANSI_COLOR_RESET);
}



void
Histogram::plotImageData(int max_id){
    if(max_id!=0){
        printf("%3i Elements: ",max_id);
    } else {
        printf("\n\n");
    }

    for(int y=0; y<height; y++){
        for(int x=0; x<width; x++){

            if(max_id>0 && max_id<=width*y+x){
                printf("\n");
                return;
            }

            if(max_id<0 && (width*height)-(width*y+x) > abs(max_id) ){
                continue;
            }

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
        if(max_id==0){
                printf("\n");
            }
    }
    printf("\n");
}


void
Histogram::calcHistCPU() {
    for (int i = 0; i < width*height*3; i+=3) {
        float r = (float)(rgb_data[i]);
        float g = (float)(rgb_data[i+1]);
        float b = (float)(rgb_data[i+2]);
        float lum = 0.2126*r + 0.7152*g + 0.0722*b;
        hist_cpu[int(lum)]++;
        // fake local hists like on gpu, step 1
        local_histograms_cpu[int(lum) + ((i/(64*32*3))*256)]++;
    }
}


void
Histogram::plotLocalHistograms(cl_uint * local_histo){
    for (int i = 0; i < workgroups; ++i) {
        int sum=0;
        printf("LOCAL HISTOGRAM: %i\n", i);
        for (int j = 0; j < 256; ++j) {
            printf("[%3i]: %12i; ",j, local_histo[i*256+j]);
            sum += local_histo[i*256+j];
            if((j+1)%10==0)
                printf("\n");
        }
        printf("\nSum: %d\n", sum);
    }
}


void
Histogram::compareGPUvsCPU(){
    for (int i = 0; i < workgroups; ++i) {
        int compare_res = memcmp(local_histograms_cpu+(i*256*sizeof(cl_uint)),
                                 local_histograms_gpu+(i*256*sizeof(cl_uint)),
                                 256*sizeof(cl_uint));
//        int compare_res = memcmp(local_histograms_cpu, local_histograms_gpu, 256*sizeof(cl_uint));
        if(compare_res==0){
            printf(ANSI_COLOR_BRIGHTGREEN);
            printf(ANSI_BOLD);
            printf("Local Histogram %3i OK!\n", i);
            printf(ANSI_COLOR_RESET);
        } else {
            int  _diff = 0;
            for (int j = 0; j < 256; ++j) {
                if(local_histograms_cpu[i*256+j] != local_histograms_gpu[i*256+j]){
                    printf(ANSI_COLOR_RED);
                    printf("Local Histogram %3i differs from CPU at %i:    ", i, j);
                    printf("CPU: %i, ", local_histograms_cpu[i*256+j]);
                    printf("GPU: %i\n", local_histograms_gpu[i*256+j]);
                    _diff++;
                    break;
                }
            }
            printf(ANSI_COLOR_RESET);

            if(_diff==0)
                printf("Local Histogram %3i OK? MEMCMP!=0 but no difference detected!\n", i);


        }
    }
}


void
Histogram::calcHistGPU(){
    cl_int status=0;

    cl_mem rgb_buffer = clCreateBuffer(OpenCLmgr->context,
                                       CL_MEM_READ_ONLY,
                                       buffersize*sizeof(u_char),
                                       nullptr,
                                       nullptr);

    status = clEnqueueWriteBuffer(OpenCLmgr->commandQueue,
                                  rgb_buffer,
                                  CL_TRUE,
                                  0,
                                  buffersize*sizeof(u_char),
                                  rgb_data,
                                  0,
                                  nullptr,
                                  nullptr);
    check_error(status);


    cl_mem all_hist_buffer = clCreateBuffer(OpenCLmgr->context,
                                        CL_MEM_READ_WRITE,
                                        workgroups*256*sizeof(cl_uint),
                                        nullptr,
                                        nullptr);

    cl_mem hist_buffer = clCreateBuffer(OpenCLmgr->context,
                                        CL_MEM_READ_WRITE,
                                        256*sizeof(cl_uint),
                                        nullptr,
                                        nullptr);

    status = clEnqueueWriteBuffer(OpenCLmgr->commandQueue,
                                  all_hist_buffer,
                                  CL_TRUE,
                                  0,
                                  workgroups*256*sizeof(cl_uint),
                                  local_histograms_gpu,
                                  0,
                                  nullptr,
                                  nullptr);
    check_error(status);


    status = clSetKernelArg(OpenCLmgr->kernels["calcStatistic_kernel"],
                            0,
                            sizeof(cl_mem),
                            (void *) &rgb_buffer );
    check_error(status);

    status = clSetKernelArg(OpenCLmgr->kernels["calcStatistic_kernel"],
                            1,
                            sizeof(uint),
                            (void *) &buffersize );
    check_error(status);

    status = clSetKernelArg(OpenCLmgr->kernels["calcStatistic_kernel"],
                            2,
                            sizeof(cl_mem),
                            (void *) &all_hist_buffer );
    check_error(status);

    status = clEnqueueNDRangeKernel(OpenCLmgr->commandQueue,
                                    OpenCLmgr->kernels["calcStatistic_kernel"],
                                    1,
                                    nullptr,
                                    gws,
                                    lws,
                                    0,
                                    nullptr,
                                    nullptr);
    check_error(status);


    status = clEnqueueReadBuffer( OpenCLmgr->commandQueue,
                                  all_hist_buffer,
                                  CL_TRUE,
                                  0,
                                  workgroups*256*sizeof(cl_uint),
                                  local_histograms_gpu,
                                  0,
                                  nullptr,
                                  nullptr);
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
                                    nullptr,
                                    lws,
                                    lws,
                                    0,
                                    nullptr,
                                    nullptr);
    check_error(status);

    status = clEnqueueReadBuffer( OpenCLmgr->commandQueue,
                                  hist_buffer,
                                  CL_TRUE,
                                  0,
                                  256*sizeof(int),
                                  this->hist,
                                  0,
                                  nullptr,
                                  nullptr);
    check_error(status);

    clReleaseMemObject(rgb_buffer);
    clReleaseMemObject(hist_buffer);
    clReleaseMemObject(all_hist_buffer);

//    delete [] local_histograms;
    // free(local_histograms);

//    plotImageData();
//    plotHistogram();
}