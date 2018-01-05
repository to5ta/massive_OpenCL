#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "OpenCLMgr.h"

void render(int *histValues)
{
    const int width = 256;
    const int height = 128;
    int max = 0;
    for (int i = 0; i < width; i++) {
        if (histValues[i] > max)
            max = histValues[i];
    }
    printf("max: %d\n", max);

    double norm = 127.0 / max;
    printf("norm: %f\n", norm);

    int8_t hist[width * height];

    // Assign values to the elements
    int value = 0;
    for (int column = 0; column < width; column++) {
        value = (int)(norm * histValues[column]);
        for (int row = 0; row < height; row++) {
            hist[row * width + column] = (int8_t)255;
            if(row >= (height - value))
                hist[row * width + column] = (int8_t)0;
        }
    }

    std::string outputImgFpn("/Users/mag/MassiveOpenCL/mag/histogram/histogram.png");
    stbi_write_png(outputImgFpn.c_str(), width, height, 1, &hist, 256 * sizeof(int8_t));
}

int main()
{
    std::string inputImgFpn("/Users/mag/MassiveOpenCL/mag/histogram/rainbow.png");
    int width, height, channels;
    unsigned char *inputImg = stbi_load(inputImgFpn.c_str(), &width, &height, &channels, 0);
    unsigned char imgArray[width * height * 4];


//  SIZE SETTINGS
    int NUM_PIXELS = height * width;
    int WORKITEM_SIZE = 256;            // wieviele pixel von einem workitem bearbeitet werden
    int WORKGROUP_SIZE = 32;            // wieviele workitems pro workgroup
    int NUM_GLOBAL_ITEMS = NUM_PIXELS / WORKITEM_SIZE;
    int PIXELS_PER_WORKGROUP = WORKITEM_SIZE * WORKGROUP_SIZE;              //Pixels per WORKGROUP;
    int GLOBAL_WORK_SIZE = (NUM_PIXELS + PIXELS_PER_WORKGROUP) / PIXELS_PER_WORKGROUP
        * WORKGROUP_SIZE;  // Isn't this one to many when NUM_PIXLES is multiple of WORKGROUP_ITEMS?

    size_t global_work_size = GLOBAL_WORK_SIZE;
    size_t local_work_size = WORKGROUP_SIZE;
    cl_int nr_workgroups = GLOBAL_WORK_SIZE / WORKGROUP_SIZE;


    // add fourth channel to image array for openCl char4
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
    stbi_image_free( inputImg );

    cl_int result[nr_workgroups * 256];

    OpenCLMgr mgr;
    cl_int status;

    cl_mem img_g = clCreateBuffer(mgr.context, CL_MEM_READ_ONLY, width * height * 4 * sizeof(cl_uchar), NULL, NULL);
    cl_mem result_g = clCreateBuffer(mgr.context, CL_MEM_READ_WRITE, nr_workgroups * 256 * sizeof(cl_int), NULL, NULL);

    status = clSetKernelArg(mgr.calc_kernel, 0, sizeof(img_g), &img_g);
    status = clSetKernelArg(mgr.calc_kernel, 1, sizeof(result_g), &result_g);

    status = clEnqueueWriteBuffer(mgr.commandQueue,
                                  img_g,
                                  CL_TRUE,
                                  0,
                                  width * height * 4 * sizeof(cl_uchar),
                                  imgArray,
                                  0,
                                  NULL,
                                  NULL);

    status = clEnqueueNDRangeKernel(mgr.commandQueue,
                                    mgr.calc_kernel,
                                    1,
                                    NULL,
                                    &global_work_size,
                                    &local_work_size,
                                    0,
                                    NULL,
                                    NULL);

    status = clSetKernelArg(mgr.reduce_kernel, 0, sizeof(result_g), &result_g);
    status = clSetKernelArg(mgr.reduce_kernel, 1, sizeof(cl_int), (void *) &nr_workgroups);

    global_work_size = 256;
    local_work_size = 256;

    status = clEnqueueNDRangeKernel(mgr.commandQueue,
                                    mgr.reduce_kernel,
                                    1,
                                    NULL,
                                    &global_work_size,
                                    &local_work_size,
                                    0,
                                    NULL,
                                    NULL);

    status = clEnqueueReadBuffer(mgr.commandQueue,
                                 result_g,
                                 CL_TRUE,
                                 0,
                                 (256 * sizeof(cl_int)),
                                 &result,
                                 0,
                                 NULL,
                                 NULL);

    render(result);

    return 0;
}