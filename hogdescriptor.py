import numpy as np
from skimage.feature import hog
import cv2

class HOGDescriptor(object):
    def __init__(self):
        self.orientations = 9 # number of gradient bins
        self.cx, self.cy = (1, 1) # pixels per cell

    def rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

    def describe(self, im):
        # convert rgb to grayscale if needed
        # if im.ndim == 3:
        #     image = self.rgb2gray(im)
        # else:
        #     image = np.at_least_2d(im)
        # sx, sy = image.shape # image size
        #
        # gx = np.zeros(image.shape)
        # gy = np.zeros(image.shape)
        # gx[:, :-1] = np.diff(image, n=1, axis=1) # compute gradient on x-direction
        # gy[:-1, :] = np.diff(image, n=1, axis=0) # compute gradient on y-direction
        # grad_mag = np.sqrt(gx ** 2 + gy ** 2) # gradient magnitude
        # grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90 # gradient orientation
        #
        # n_cellsx = int(np.floor(sx / self.cx))  # number of cells in x
        # n_cellsy = int(np.floor(sy / self.cy))  # number of cells in y
        # # compute orientations integral images
        # orientation_histogram = np.zeros((n_cellsx, n_cellsy, self.orientations))
        # for i in range(self.orientations):
        #     # create new integral image for this orientation
        #     # isolate orientations in this range
        #     temp_ori = np.where(grad_ori < 180 / self.orientations * (i + 1),
        #                 grad_ori, 0)
        #     temp_ori = np.where(grad_ori >= 180 / self.orientations * i,
        #                 temp_ori, 0)
        #     # select magnitudes for those orientations
        #     cond2 = temp_ori > 0
        #     temp_mag = np.where(cond2, grad_mag, 0)
        #     orientation_histogram[:,:,i] = uniform_filter(temp_mag, size=(self.cx, self.cy))[self.cx/2::self.cx, self.cy/2::self.cy]
        #
        # return orientation_histogram.ravel()
        if im.ndim == 3:
            image = self.rgb2gray(im)
        else:
            image = np.at_least_2d(im)
        fd, hog_image = hog(image, orientations = self.orientations, pixels_per_cell=(128, 128),
                    cells_per_block=(self.cx,self.cy), visualise=True)
        return fd

# hogd = HOGDescriptor()
# image = cv2.imread("./dataset/100502.jpg")
# print len(hogd.describe(image))
