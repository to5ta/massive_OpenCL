import pyopencl as cl
import numpy as np


def calcHistogram(inputImg):
    # SIZE SETTINGS
    NUM_PIXELS = inputImg.shape[0] * inputImg.shape[1]
    WORKITEM_SIZE = 256
    WORKGROUP_SIZE = 32
    NUM_GLOBAL_ITEMS = NUM_PIXELS / WORKITEM_SIZE
    WORKGOUP_ITEMS = WORKITEM_SIZE * WORKGROUP_SIZE  # Pixels per WORKGROUP
    GLOBAL_WORK_SIZE = (NUM_PIXELS + WORKGOUP_ITEMS) / WORKGOUP_ITEMS * WORKGROUP_SIZE  # Isn't this one to many when NUM_PIXLES is multiple of WORKGROUP_ITEMS?

    # Get platforms, both CPU and GPU
    plat = cl.get_platforms()
    CPU = plat[0].get_devices()
    try:
        GPU = plat[1].get_devices()
    except IndexError:
        GPU = "none"

    GPU = "none"
    #Create context for GPU/CPU
    if GPU!= "none":
        ctx = cl.Context(GPU)
    else:
        ctx = cl.Context(CPU)

    # Create queue for each kernel execution
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags

    # Kernel function
    with open('histogramKernel.cl', 'r') as kernel:
        src = kernel.read().replace('\n', '')

    #Kernel function instantiation
    prg = cl.Program(ctx, src).build()

    # Flatten image so it can be read in kernel as float4
    img = inputImg.reshape(NUM_PIXELS*3)

    #Allocate memory for variables on the device
    img_g =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
    result_g = cl.Buffer(ctx, mf.WRITE_ONLY, (256 * np.dtype(np.uint32).itemsize))

    # Create Kernel.
    krnl = prg.histogram

    kernel.set_arg(0, img_g)
    kernel.set_arg(1, result_g)

    # Array to copy result into
    result = np.zeros([256], dtype=np.uint32)


    global_work_size = GLOBAL_WORK_SIZE
    nr_workgroups = global_work_size / WORKGROUP_SIZE
    local_work_size = WORKGROUP_SIZE

    cl.enqueue_nd_range_kernel(queue, kernel, [global_work_size], local_work_size, global_work_offset=None,
                                     wait_for=None, g_times_l=False)
    cl.enqueue_copy(queue, result, result_g)

    return result