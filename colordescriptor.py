import cv2
import numpy as np

class ColorDescriptor(object):
    def __init__(self, bins):
        self.bins = bins

    def describe(self, image):
        #Return features
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []

        #get dimensions and center of the image
        (h, w) = image.shape[:2]
        (centerX, centerY) = (int(w * 0.5), int(h * 0.5))

        #divide the image into 4 blocks: top-left, top-right, bottom-left, bottom-right
        segments = [(0, centerX, 0, centerY), (centerX, w, 0, centerY), (centerX, w, centerY, h), (0, centerX, centerY, h)]

        #create an elliptical mask to represent the image center.
        (axesX, axesY) = (int(w * 0.75) / 2, int(h * 0.75) / 2)
        ellipMask = np.zeros(image.shape[:2], dtype = 'uint8')
        cv2.ellipse(ellipMask, (centerX, centerY), (axesX, axesY), 0, 0, 360, 255, -1)

        #loop over blocks
        for (startX, endX, startY, endY) in segments:
            #create a mask for each block of the image.
            cornerMask = np.zeros(image.shape[:2], dtype = 'uint8')
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)

            #extract color histogram from the block of the image, update feature vector
            hist = self.getHistogram(image, cornerMask)
            features.extend(hist)

        #extract color histogram from elliptical center and update feature vector
        hist = self.getHistogram(image, ellipMask)
        features.extend(hist)

        return features


    def getHistogram(self, image, mask):
        #extract 3D color histogram from the masked block of the image, using the number of bins per channel.
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])

        #normalize the histogram
        hist = cv2.normalize(hist, hist).flatten()

        return hist

#cd = ColorDescriptor((8, 12, 3))
#image = cv2.imread("./visa.jpg")
#print cd.describe(image)
