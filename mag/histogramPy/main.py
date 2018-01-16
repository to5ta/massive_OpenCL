import os
import numpy as np
from PIL import Image

import histogramPy
import histogramOCL
import pyopencl_tests
import testKernel
import pyopencl.array as cl_array

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


def showHist(vals):
    # if len(vals) > 256:
    #     for i in xrange(256):
    #         reducedValue = 0
    #         for k in xrange(len(vals)/256):
    #             reducedValue += vals[k * 256 + i]
    #         vals[i] = reducedValue

    vals[0] = 0     # values for 0 are far to high!?
    vals = vals[:256]

    max = vals.max()
    norm = 128.0 / max

    hist = np.ones((128, 256), dtype=np.uint8)
    hist = hist * 255

    for x in xrange(256):
        val = int(norm * vals[x])
        if val > 125:
            print val

        hist[(128 - val):128, x] = 0

    # Show the image
    imgOut = Image.fromarray(hist)
    print imgOut.show()

#Read in image
img = Image.open('/Users/mag/parallelComputing/MassiveOpenCL/mag/histogramPy/resources/gradient_1.png')
npImg = np.asarray(img)

# pyopencl_tests.showDevices()
# vals = histogramPy.calcHistogram(npImg)

vals = histogramOCL.calcHistogram(npImg)
try:
    showHist(vals)
except:
    print 'cant show histogram'
