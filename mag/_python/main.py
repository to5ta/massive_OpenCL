import os

import numpy as np
from PIL import Image

import histogramPy
import histogramOCL
import pyopencl.array as cl_array



os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'




def test(img):

    # testArray = np.array(([1, 2, 3], [1, 2, 3]))
    testArray = np.array( ([[0, 0, 0], [1, 1, 1], [2,2,2], [3,3,3]],
                           [[4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]]) )
    print testArray.shape

    # testArray = testArray.transpose(2, 0, 1)
    testArray = testArray.reshape(-1, testArray.shape[1])

    # img = img.reshape(-1, testArray.shape[1])

    print testArray.shape
    print testArray






def showHist(vals):
    max = vals.max()

    hist = np.ones((128, 256), dtype=np.int8)
    hist = hist * 255

    for x in xrange(256):
        val = int(127.0 / max * vals[x])

        hist[(128 - val):127, x] = 0

    # Show the blurred image
    imgOut = Image.fromarray(hist)
    print imgOut.show()




#Read in image
img = Image.open('rainbow.png')
npImg = np.asarray(img)
# test(npImg)
# # vals = histogramPy.calcHistogram(npImg)
vals = histogramOCL.calcHistogram(npImg)
try:
    showHist(vals)
except:
    print 'cant show histogram'
