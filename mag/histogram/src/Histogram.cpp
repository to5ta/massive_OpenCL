#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "OpenCLMgr.h"
#include "clstatushelper.h"


void render(int *histValues)
{
    const int width = 256;
    const int height = 128;
    int max = 0;
    for (int i = 0; i < width; i++) {
        if (histValues[i] > max)
            max = histValues[i];
    }
//    printf("max: %d\n", max);

    double norm = 127.0 / max;
//    printf("norm: %f\n", norm);

    int8_t hist[width * height];

    // Assign values to the elements
    int value = 0;
    for (int column = 0; column < width; column++) {
        value = (int) (norm * histValues[column]);
        for (int row = 0; row < height; row++) {
            hist[row * width + column] = (int8_t) 255;
            if (row >= (height - value))
                hist[row * width + column] = (int8_t) 0;
        }
    }

    std::string outputImgFpn("/Users/mag/parallelComputing/MassiveOpenCL/mag/histogram/resources/histogram.png");
    stbi_write_png(outputImgFpn.c_str(), width, height, 1, &hist, 256 * sizeof(int8_t));
}

double calc()
{
    std::string inputImgFpn("/Users/mag/parallelComputing/MassiveOpenCL/mag/histogram/resources/rainbow.png");
    int width, height, channels;
    unsigned char *inputImg = stbi_load(inputImgFpn.c_str(), &width, &height, &channels, 0);
    unsigned char imgArray[width * height * 4];


//  SIZE SETTINGS
    cl_int PIXELS = pow(2, 11);

    int NUM_PIXELS = height * width;
    int WORKITEM_SIZE = PIXELS;            // wieviele pixel von einem workitem bearbeitet werden
    int WORKGROUP_SIZE = 32;            // wieviele workitems pro workgroup
    int NUM_GLOBAL_ITEMS = NUM_PIXELS / WORKITEM_SIZE;
    int PIXELS_PER_WORKGROUP = WORKITEM_SIZE * WORKGROUP_SIZE;              //Pixels per WORKGROUP;
    int GLOBAL_WORK_SIZE = (NUM_PIXELS + (PIXELS_PER_WORKGROUP-1)) / PIXELS_PER_WORKGROUP * WORKGROUP_SIZE;

    size_t global_work_size = GLOBAL_WORK_SIZE;
    size_t local_work_size = WORKGROUP_SIZE;
    cl_int nr_workgroups = GLOBAL_WORK_SIZE / WORKGROUP_SIZE;


    // add fourth channel to image array for openCl char4
    if (channels == 3) {
        int idx = 0;
        for (int i = 0; i < width * height * 3; i++) {
            if ((idx + 1) % 4 == 0) {
                imgArray[idx] = 1;
                idx += 1;
                i -= 1;
            }
            else {
                imgArray[idx] = inputImg[i];
                idx += 1;
            }
        }
    }
    else if (channels == 4) {
        for (int i = 0; i < width * height * 4; i++) {
            imgArray[i] = inputImg[i];
        }
    }
    stbi_image_free(inputImg);

    cl_int result[nr_workgroups * 256];

    OpenCLMgr mgr;
    cl_int status;
    cl_event eventCalc;
    cl_event eventReduce;
    cl_ulong time_start_calc;
    cl_ulong time_start_reduce;
    cl_ulong time_end_calc;
    cl_ulong time_end_reduce;

    cl_mem img_g = clCreateBuffer(mgr.context,
                                  CL_MEM_READ_ONLY,
                                  width * height * 4 * sizeof(cl_uchar),
                                  NULL,
                                  NULL);
    cl_mem result_g = clCreateBuffer(mgr.context,
                                     CL_MEM_READ_WRITE,
                                     nr_workgroups * 256 * sizeof(cl_int),
                                     NULL,
                                     NULL);


    status = clSetKernelArg(mgr.calc_kernel, 0, sizeof(img_g), &img_g);
    status = clSetKernelArg(mgr.calc_kernel, 1, sizeof(result_g), &result_g);
    check_error(status);

    status = clSetKernelArg(mgr.calc_kernel, 2, sizeof(cl_int), (void *) &PIXELS);

    // WRITE BUFFER
    status = clEnqueueWriteBuffer(mgr.commandQueue,
                                  img_g,
                                  CL_TRUE,
                                  0,
                                  width * height * 4 * sizeof(cl_uchar),
                                  imgArray,
                                  0,
                                  NULL,
                                  NULL);


    // ENQUEUE KERNELS
    status = clEnqueueNDRangeKernel(mgr.commandQueue,
                                    mgr.calc_kernel,
                                    1,
                                    NULL,
                                    &global_work_size,
                                    &local_work_size,
                                    0,
                                    NULL,
                                    &eventCalc);


    // READ BUFFER
    status = clEnqueueReadBuffer(mgr.commandQueue,
                                 result_g,
                                 CL_TRUE,
                                 0,
                                 (256 * nr_workgroups * sizeof(cl_int)),
                                 &result,
                                 0,
                                 NULL,
                                 NULL);


    check_error(status);
    status = clWaitForEvents(1, &eventCalc);
    check_error(status);

    // READ PROFILING START
    status = clGetEventProfilingInfo(eventCalc,
                                     CL_PROFILING_COMMAND_START,
                                     sizeof(time_start_calc),
                                     &time_start_calc,
                                     NULL);
    check_error(status);
    // READ PROFILING END
    status = clGetEventProfilingInfo(eventCalc,
                                     CL_PROFILING_COMMAND_END,
                                     sizeof(time_end_calc),
                                     &time_end_calc,
                                     NULL);
    check_error(status);

    double nanoSeconds = time_end_calc - time_start_calc;
    double miliSecondsCalc = nanoSeconds / 1000000.0;
    printf("OpenCl Execution time is: %0.3f milliseconds \n", (miliSecondsCalc));


    status = clSetKernelArg(mgr.reduce_kernel, 0, sizeof(result_g), &result_g);
    status = clSetKernelArg(mgr.reduce_kernel, 1, sizeof(cl_int), (void *) &nr_workgroups);

    global_work_size = 256;
    local_work_size = 256;

    // WRITE BUFFER
    status = clEnqueueWriteBuffer(mgr.commandQueue,
                                  result_g,
                                  CL_TRUE,
                                  0,
                                  nr_workgroups * 256 * sizeof(cl_int),
                                  result,
                                  0,
                                  NULL,
                                  NULL);
    check_error(status);

    // ENQUEUE KERNELS
    status = clEnqueueNDRangeKernel(mgr.commandQueue,
                                    mgr.reduce_kernel,
                                    1,
                                    NULL,
                                    &global_work_size,
                                    &local_work_size,
                                    0,
                                    NULL,
                                    &eventReduce);
    check_error(status);

    status = clWaitForEvents(1, &eventReduce);
    check_error(status);

    // READ BUFFER
    status = clEnqueueReadBuffer(mgr.commandQueue,
                                 result_g,
                                 CL_TRUE,
                                 0,
                                 (256 * sizeof(cl_int)),
                                 &result,
                                 0,
                                 NULL,
                                 NULL);
    check_error(status);

    // READ PROFILING START
    status = clGetEventProfilingInfo(eventReduce,
                                     CL_PROFILING_COMMAND_START,
                                     sizeof(time_start_reduce),
                                     &time_start_reduce,
                                     NULL);
    // READ PROFILING END
    status = clGetEventProfilingInfo(eventReduce,
                                     CL_PROFILING_COMMAND_END,
                                     sizeof(time_end_reduce),
                                     &time_end_reduce,
                                     NULL);

    nanoSeconds = time_end_reduce - time_start_reduce;
    double miliSeconds = nanoSeconds / 1000000.0;
    printf("OpenCl Execution time is: %0.3f milliseconds \n", (miliSeconds));

    status = clFinish(mgr.commandQueue);
    check_error(status);
    status = clReleaseMemObject(img_g);
    check_error(status);
    status = clReleaseMemObject(result_g);
    check_error(status);

    render(result);
    return (miliSeconds + miliSecondsCalc);
}

int main()
{
    const int ITERATIONS = 10;
    double sum = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        sum += calc();
    }
    printf("%f\n", sum / ITERATIONS);
}