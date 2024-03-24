
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include "OpenCLMgr.h"
#include "clstatushelper.h"

#define DEBUG_PRINT 0


using namespace std;


OpenCLMgr::OpenCLMgr(cl_command_queue_properties queue_flags) {
    context = 0;
    commandQueue = 0;
    program = 0;

    valid = (init(queue_flags) == SUCCESS);
}


OpenCLMgr::~OpenCLMgr() {
    cl_int status;

    //Release kernels
//	if (matadd_kernel) status = clReleaseKernel(matadd_kernel);


    //Release the program object.
    if (program) status = clReleaseProgram(program);

    //Release  Command queue.
    if (commandQueue) status = clReleaseCommandQueue(commandQueue);

    //Release context.
    if (context) status = clReleaseContext(context);
}



cl_int
OpenCLMgr::setVariable(const char* DEF_NAME,
                       const int value){

    size_t length_keyword = strlen(DEF_NAME);
    size_t length_source = strlen(source);

    char new_define[20];

    int lendef = sprintf(new_define, " %-5i", value);
//    printf("%s: '%s'\n",DEF_NAME, new_define);

    for(int i=0; i<length_source-length_keyword-1; i++){
        if(0==memcmp(source+i, DEF_NAME, length_keyword)){
//            printf("\n\nFOUND: %s \n\n, %i", source+i, i);

            printf("'");

#if(DEBUG_PRINT)
            for (int k = i-8; k < i+30; ++k) {
                printf("%c", source[k]);
            }
            printf("' ->\n");
#endif

            for (int j = 0; j < lendef; ++j) {
                source[i+length_keyword+j] = new_define[j];
            }

#if(DEBUG_PRINT)
            printf("'");

            for (int k = i-8; k < i+30; ++k) {
                printf("%c", source[k]);
            }
            printf("'\n");
#endif
            break;

        }
    }


}




/* convert the kernel file into a string */
int OpenCLMgr::convertToString(const char *filename, std::string &s) {
    size_t size;
    char *str;
    std::fstream f(filename, (std::fstream::in | std::fstream::binary));

    if (f.is_open()) {
        size_t fileSize;
        f.seekg(0, std::fstream::end);
        size = fileSize = (size_t) f.tellg();
        f.seekg(0, std::fstream::beg);
        str = new char[size + 1];
        if (!str) {
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
    cout << "Error: failed to open file\n:" << filename << endl;
    return FAILURE;
}


cl_int OpenCLMgr::init(cl_command_queue_properties queue_flags) {
    cl_int status;
    status = createContext(queue_flags);
    check_error(status);
    return status;
}


cl_int OpenCLMgr::createContext(cl_command_queue_properties queue_flags) {
    deviceNo = 1;

    // Getting platforms and choose an available one.
    //the NO. of platforms
    cl_uint numPlatforms = 0;
    //the chosen platform
    cl_platform_id platform = NULL;
    cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
    check_error(status);
    CHECK_SUCCESS("Error: Getting platforms!")

    // For clarity, choose the first available platform.
    if (numPlatforms > 0) {
        cout << "Found " << numPlatforms << " platforms." << endl;
        cl_platform_id *platforms = (cl_platform_id *) malloc(numPlatforms * sizeof(cl_platform_id));
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        check_error(status);
        platform = platforms[0];
        free(platforms);
        CHECK_SUCCESS("Error: Getting platforms ids")
    }

    // Query devices and choose a GPU device if has one. Otherwise use the CPU as device.*/
    numDevices = 0;
//	cl_device_id *devices;
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    check_error(status);
    CHECK_SUCCESS("Error: Getting device ids")
    if (numDevices == 0)    //no GPU available.
    {
        cout << "No GPU device available." << endl;
        cout << "Choose CPU as default device." << endl;
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
        CHECK_SUCCESS("Error: Getting number of cpu devices")
        check_error(status);
        devices = (cl_device_id *) malloc(numDevices * sizeof(cl_device_id));
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
        CHECK_SUCCESS("Error: Getting cpu device id")
        check_error(status);
    } else {
        devices = (cl_device_id *) malloc(numDevices * sizeof(cl_device_id));
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
        CHECK_SUCCESS("Error: Getting gpu device id")
        check_error(status);
    }

    if (deviceNo >= numDevices)
        deviceNo = 0;


    // only information
    char devname[100] = {0};
    clGetDeviceInfo(devices[deviceNo], CL_DEVICE_NAME, 100, devname, NULL);
    cout << "Using Device: " << devname << endl;

    // store size of memory which can be allocated at once
    clGetDeviceInfo(devices[deviceNo], CL_DEVICE_MAX_MEM_ALLOC_SIZE, 100, &(maxMem), NULL);


//	cout << "CL_DEVICE_MAX_MEM_ALLOC_SIZE: " << maxMem << " bytes" << endl;
//	cout << "CL_DEVICE_MAX_MEM_ALLOC_SIZE: " << maxMem / 1048576.f << " MB" << endl;
//
//
//	cl_uint devMaxComputeUnits = 0;
//	clGetDeviceInfo(devices[deviceNo], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(devMaxComputeUnits), &(devMaxComputeUnits),
//					NULL);
//	cout << "CL_DEVICE_MAX_COMPUTE_UNITS: " << devMaxComputeUnits << endl;
//
    size_t devMaxWorkGroupSize = 0;
    clGetDeviceInfo(devices[deviceNo], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(devMaxWorkGroupSize),
                    &(devMaxWorkGroupSize), NULL);
    cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << devMaxWorkGroupSize << endl;
//
//	cl_uint devMaxWorkItemDims = 0;
//	clGetDeviceInfo(devices[deviceNo], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(devMaxWorkItemDims),
//					&(devMaxWorkItemDims), NULL);
//	cout << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << devMaxWorkItemDims << endl;
//	cl_uint devMaxWorkItemDims = 0;
//	clGetDeviceInfo(devices[deviceNo], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(devMaxWorkItemDims), &(devMaxWorkItemDims), NULL);
//	cout << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << devMaxWorkItemDims << endl;
//
//	size_t maxPerDims[3] = {0};
//	clGetDeviceInfo(devices[deviceNo], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * 3, &(maxPerDims), NULL);
//	cout << "CL_DEVICE_MAX_WORK_ITEM_SIZES: " << maxPerDims[0] << ", " << maxPerDims[1] << ", " << maxPerDims[2]
//		 << endl;
//
//	CL_DEVICE_MAX_WORK_ITEM_SIZES
//	CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
//	CL_DEVICE_MAX_WORK_ITEM_SIZES
//	CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
//  .....


    // Create context
    context = clCreateContext(NULL, 1, devices + deviceNo, NULL, NULL, NULL);
    CHECK_SUCCESS("Error: creating OpenCL context")

//	char platforminfo[100] = {0};
//	clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 100, platforminfo, NULL);
//	cout << "PlatformInfo: "<< platforminfo << endl;

    // Creating command queue associate with the context

//	commandQueue = clCreateCommandQueue(context, devices[deviceNo], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &status);
    commandQueue = clCreateCommandQueue(context, devices[deviceNo], queue_flags, &status);
//    clCreateCommandQueueWithProperties(context, devices[deviceNo], )
    CHECK_SUCCESS("Error: creating command queue")


    return status;

}




cl_int OpenCLMgr::loadFile(const char *filepath) {
    cl_int status;

//	const char *filepath = "../Matrix_Kernel.cl";

    status = convertToString(filepath, sourceStr);
    CHECK_SUCCESS("Error: loading OpenCL file")

    source = (char*)(sourceStr.c_str());

    return SUCCESS;
}



cl_int OpenCLMgr::buildProgram() {

    cl_int status;
    size_t sourceSize[] = {strlen(source)};

    const char* temp = (const char*)(source);
    program = clCreateProgramWithSource(context, 1, &temp, sourceSize, &status);
    CHECK_SUCCESS("Error: creating OpenCL program")

    // Build program.
    status = clBuildProgram(program, 1, devices + deviceNo, NULL, NULL, NULL);
    if (status) {
        char msg[120000];
        clGetProgramBuildInfo(program, devices[deviceNo], CL_PROGRAM_BUILD_LOG, sizeof(msg), msg, NULL);
        cerr << "=== build failed ===\n" << msg << endl;
        getc(stdin);
        return FAILURE;
    }
    return status;
}


cl_int OpenCLMgr::createKernels(const char **kernel_names, cl_uint count) {

    cl_int status;
    for (cl_int i = 0; i < count; i++) {
        // Create kernel objects
        kernels[kernel_names[i]] = clCreateKernel(program, kernel_names[i], &status);
        CHECK_SUCCESS("Error: creating '" << kernel_names[i] << "' kernel")
        check_error(status);
    }

    // this may cause problems if program was not built and kernels not created yet?
    if (devices != NULL) {
        free(devices);
        devices = NULL;
    }

    return status;


}







