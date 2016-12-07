from colordescriptor import ColorDescriptor
import argparse
import glob
import cv2

#create argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "Path to directory of image datasets.")
ap.add_argument("-ci", "--cindex", required = True, help = "Path to the index storage.")
args = vars(ap.parse_args())

#initialize color descriptor
cd = ColorDescriptor((8, 12, 3))

#open the index file
indexFile = open(args["cindex"], 'w')

for imagePath in glob.glob(args["dataset"] + "/*.jpg"):
    UID = imagePath[imagePath.rfind('/') + 1:]
    image = cv2.imread(imagePath)

    #describe the image
    features = cd.describe(image)

    #output features to index file
    features = [str(f) for f in features]
    indexFile.write("%s, %s\n" % (UID, ",".join(features)))

#close index file
indexFile.close()
