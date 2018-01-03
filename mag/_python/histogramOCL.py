import pyopencl as cl
import numpy as np
from time import time



def calcHistogram(inputImg):
    # SIZE SETTINGS
    NUM_PIXELS = inputImg.shape[0] * inputImg.shape[1]
    WORKITEM_SIZE = 256     # wieviele pixel von einem workitem bearbeitet werden
    WORKGROUP_SIZE = 32     # wieviele workitems pro workgroup
    NUM_GLOBAL_ITEMS = NUM_PIXELS / WORKITEM_SIZE
    PIXELS_PER_WORKGROUP = WORKITEM_SIZE * WORKGROUP_SIZE  # Pixels per WORKGROUP
    GLOBAL_WORK_SIZE = (NUM_PIXELS + PIXELS_PER_WORKGROUP) / PIXELS_PER_WORKGROUP * WORKGROUP_SIZE  # Isn't this one to many when NUM_PIXLES is multiple of WORKGROUP_ITEMS?

    global_work_size = GLOBAL_WORK_SIZE
    nr_workgroups = GLOBAL_WORK_SIZE / WORKGROUP_SIZE

    print "nr_workgroups:", nr_workgroups
    print "global_work_size:", global_work_size
    print "NUM_GLOBAL_ITEMS:", NUM_GLOBAL_ITEMS


    local_work_size = WORKGROUP_SIZE

    # Get platforms, both CPU and GPU
    plat = cl.get_platforms()[0]

    CPU = plat.get_devices(device_type=cl.device_type.CPU)[0]
    try:
        GPU = plat.get_devices(device_type=cl.device_type.GPU)[0]
    except IndexError:
        GPU = 'none'

    # GPU = 'none'
    # Create context for GPU/CPU
    try:
        if GPU._id == "device":
            ctx = cl.Context([GPU])
            print 'Created GPU Context'
    except:
        ctx = cl.Context([CPU])
        print 'Created CPU Context'

    # Create queue for each kernel execution
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    mf = cl.mem_flags

    # Kernel function
    with open('calcStatistic.cl', 'r') as kernelCalc:
        srcCalc = kernelCalc.read()


    # Kernel function instantiation
    prg = cl.Program(ctx, srcCalc)
    prg = prg.build()

    # Flatten image so it can be read in kernel as uchar4
    img = np.ones((256, 512, 3 + 1), dtype=np.uint8)
    img[:, :,  :-1] = inputImg
    img = img.reshape(NUM_PIXELS * 4)


    # Allocate memory for variables on the device
    img_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
    result_g = cl.Buffer(ctx, mf.WRITE_ONLY, (nr_workgroups * 256 * np.dtype(np.int32).itemsize))


    # Array to copy result into
    result = np.zeros([nr_workgroups * 256], dtype=np.int32)

    print "global_work_size:", global_work_size
    print "local_work_size:", local_work_size

    gpu_start_time = time()  # Get the GPU start time

    event = prg.calcStatistic(queue, [global_work_size], [local_work_size], img_g, result_g)  # Enqueue the GPU sum program

    event.wait()  # Wait until the event finishes
    elapsed = 1e-9 * (event.profile.end - event.profile.start)  # Calculate the time it took to execute the kernel
    print("GPU Kernel Time: {0} s".format(elapsed))  # Print the time it took to execute the kernel

    # cl.enqueue_copy(queue, result, result_g).wait() # Read back the data from GPU memory into array c_gpu

    gpu_end_time = time()  # Get the GPU end time
    print("GPU Time: {0} s".format(gpu_end_time - gpu_start_time))  # Print the time the GPU program took, including both memory copies

##########################################################################################
##########################################################################################

    with open('reduceStatistic.cl', 'r') as kernelReduce:
        srcReduce = kernelReduce.read()


    # Kernel function instantiation
    prg2 = cl.Program(ctx, srcReduce)
    prg2 = prg2.build()

    gpu_start_time = time()  # Get the GPU start time

    nWG = np.zeros(1, dtype=np.int8)
    nWG[0] = nr_workgroups
    nWG_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nWG)

    event = prg2.reduceStatistic(queue, [256], [256], result_g, nWG_g)  # Enqueue the GPU sum program

    event.wait()  # Wait until the event finishes
    elapsed = 1e-9 * (event.profile.end - event.profile.start)  # Calculate the time it took to execute the kernel
    print("GPU Kernel Time: {0} s".format(elapsed))  # Print the time it took to execute the kernel

    cl.enqueue_copy(queue, result, result_g).wait()  # Read back the data from GPU memory into array c_gpu

    gpu_end_time = time()  # Get the GPU end time
    print("GPU Time: {0} s".format(gpu_end_time - gpu_start_time))  # Print the time the GPU program took, including both memory copies

    # # Create Kernel.
    # kernel = prg.calcStatistic
    # kernel.set_arg(0, img_g)
    # kernel.set_arg(1, result_g)

    # cl.enqueue_nd_range_kernel(queue, kernel, [global_work_size], [local_work_size], global_work_offset=None,
    #                                  wait_for=None, g_times_l=False)
    # cl.enqueue_copy(queue, result, result_g)


    result = result[:256]
    return result