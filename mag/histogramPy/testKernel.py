import pyopencl as cl
import numpy as np


def test():
    # Get platforms, both CPU and GPU
    plat = cl.get_platforms()[0]

    CPU = plat.get_devices(device_type=cl.device_type.CPU)[0]
    try:
        GPU = plat.get_devices(device_type=cl.device_type.GPU)[0]
    except IndexError:
        GPU = 'none'

    GPU = 'none'
    #Create context for GPU/CPU
    try:
        if GPU._id == "device":
            ctx = cl.Context([GPU])
            print 'Created GPU Context'
    except:
        ctx = cl.Context([CPU])
        print 'Created CPU Context'

    # Create queue for each kernel execution
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags

    # Kernel function
    with open('testKernel.cl', 'r') as kernel:
        src = kernel.read()


    # Kernel function instantiation
    prg = cl.Program(ctx, src)
    prg = prg.build()


    img = np.ones((262144 * 4), np.int8)

    # Allocate memory for variables on the device
    img_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
    result_g = cl.Buffer(ctx, mf.WRITE_ONLY, (262144 * np.dtype(np.int8).itemsize))


    # Create Kernel.
    kernel = prg.test

    kernel.set_arg(0, img_g)
    kernel.set_arg(1, result_g)

    print dir(prg)
    print prg.source

    # Array to copy result into
    result = np.zeros([262144], dtype=np.int8)


    cl.enqueue_nd_range_kernel(queue, kernel, [262144], [64], global_work_offset=None,
                                     wait_for=None, g_times_l=False)
    cl.enqueue_copy(queue, result, result_g)


    return result