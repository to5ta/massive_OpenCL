#include <OpenCL/cl.h>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

#define FAILURE 1
#define SUCCESS 0

#include "ImageFX.h"
#include "OpenCLMgr.h"

/* convert the kernel file into a string */
int convertToString(const char *filename, std::string& s) {
    size_t size;
    char*  str;
    std::fstream f(filename, (std::fstream::in | std::fstream::binary));

    if(f.is_open()) {
        size_t fileSize;
        f.seekg(0, std::fstream::end);
        size = fileSize = (size_t)f.tellg();
        f.seekg(0, std::fstream::beg);
        str = new char[size+1];
        if(!str) {
            f.close();
            return 0;
        }

        f.read(str, fileSize);
        f.close();
        str[size] = '\0';
        s = str;
        delete[] str;
        return 0;
    }
    cout<<"Error: failed to open file\n:"<<filename<<endl;
    return FAILURE;
}

// ==========================
// ==== filenames for UI ====
// ==========================

std::string inputFilename = "Banane.png";
std::string outputFilename = "ImageFX.bmp";

struct ImgFXWindow : Window {
    cl_mem inmem, posmem, velmem, outmem, parmem;
    cl_kernel procKernel, clearKernel;

    int32_t *outbuf;
    SDL_Surface *image;

	OpenCLMgr *mgr;

    ImgFXWindow() {
		inmem = posmem = velmem = outmem = parmem=0;
		outbuf = 0;

		mgr = new OpenCLMgr();

		procKernel = mgr->procKernel;
		clearKernel = mgr->clearKernel;

        onReset();
    }

	~ImgFXWindow()
	{
		delete mgr;
	}

    virtual void onReset() {
		// =================================================
		// ==== called when clicking the "reset" button ====
		// =================================================
        if (inmem) clReleaseMemObject(inmem);
        if (posmem) clReleaseMemObject(posmem);
        if (velmem) clReleaseMemObject(velmem);
        if (outmem) clReleaseMemObject(outmem);
        delete[] outbuf; outbuf = 0;

		image = loadImageRGB32(inputFilename);
        if (!image) throw "loading failed";

		// create buffer for input image
        int w = image->w, h = image->h;
        inmem = clCreateBuffer(mgr->context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                               sizeof(int32_t)*w*h, image->pixels, NULL);

		{
			// additional buffers if needed
			float *pos = new float[2*w*h];
			for (int y=0; y<h; y++) {
				for (int x=0; x<w; x++) {
					pos[(y*w+x)*2] = x;
					pos[(y*w+x)*2+1] = y;
				}
			}
			posmem = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
									sizeof(float)*2*w*h, pos, NULL);
			delete[] pos;
			float *vel = new float[2*w*h];
			for (int i=0; i<2*w*h; i++)
				vel[i] = (((float)rand())/RAND_MAX-.5f)/2;
			velmem = clCreateBuffer(mgr->context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(float)*2*w*h, vel, NULL);
			delete[] vel;
		}

        outmem = clCreateBuffer(mgr->context, CL_MEM_WRITE_ONLY, sizeof(int32_t)*w*h, NULL, NULL);

		// create buffer for numeric parameters
        int32_t values[8]; memset(values, 0, sizeof(int32_t)*8);
        readValues(values, 8);
        parmem = clCreateBuffer(mgr->context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                sizeof(int32_t)*8, values, NULL);

        cl_int status;
        status = clSetKernelArg(procKernel, 0, sizeof(inmem), &inmem);
        status |= clSetKernelArg(procKernel, 1, sizeof(posmem), &posmem);
        status |= clSetKernelArg(procKernel, 2, sizeof(velmem), &velmem);
        status |= clSetKernelArg(procKernel, 3, sizeof(outmem), &outmem);
        status |= clSetKernelArg(procKernel, 4, sizeof(parmem), &parmem);

        status |= clSetKernelArg(clearKernel, 0, sizeof(outmem), &outmem);
        if (status) throw "set kernel arg";

		// copy input to output for first display
        outbuf = new int32_t[w*h];
        memcpy(outbuf, image->pixels, sizeof(int32_t)*w*h);
        useBuffer(outbuf, w, h);
    }

    virtual void onNewValues() {
		// =====================================================
		// ==== called when numeric parameters have changed ====
		// =====================================================
        int32_t values[8]; memset(values, 0, sizeof(int32_t)*8);
        readValues(values, 8);
        cl_int status = clEnqueueWriteBuffer(mgr->commandQueue, parmem, CL_TRUE, 0,
                                             sizeof(int32_t)*8, values, 0, NULL, NULL);
        if (status) throw "write buffer";
        onRender();
    }

	virtual void onSave() {
		// ================================================
		// ==== called when clicking the "save" button ====
		// ================================================
		saveBMP(outputFilename);
	}

    virtual void onRender() {
		// ==================================================
		// ==== called when clicking the "render" button ====
		// ==================================================
        if (!outbuf || !image) return;

        size_t gdims[] = { image->w, image->h };
        cl_int status;
        status = clEnqueueNDRangeKernel(mgr->commandQueue, clearKernel, 2, NULL, gdims, NULL, 0, NULL, NULL);
        status |= clEnqueueNDRangeKernel(mgr->commandQueue, procKernel, 2, NULL, gdims, NULL, 0, NULL, NULL);
        status |= clEnqueueReadBuffer(mgr->commandQueue, outmem, CL_TRUE, 0, sizeof(*outbuf)*image->w*image->h, outbuf, 0, NULL, NULL);
        if (status) throw "enqueue commands";
    }
};

int main(int argc, char *argv[]) {


	// ====== show the UI
    ImgFXWindow w;
    w.run();


    return SUCCESS;
}

//#include "stdafx.h"
//
//int _tmain(int argc, char* argv[])
//{
//	main(argc, NULL);
//	return 0;
//}
//
