import numpy as np

def calcHistogram(img):

    rows = img.shape[0]
    columns = img.shape[1]
    channels = img.shape[2]

    hist = np.zeros(256, dtype=np.int32)

    for row in xrange(rows):
        for column in xrange(columns):
            pixel = img[row, column, :]
            Y = int(0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2])
            hist[Y] = hist[Y] + 1
    return hist

