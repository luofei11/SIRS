from hogdescriptor import HOGDescriptor
import argparse
import glob
import cv2

#create argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "Path to directory of image datasets.")
ap.add_argument("-hi", "--hindex", required = True, help = "Path to the index storage.")
args = vars(ap.parse_args())

#initialize color descriptor
hogd = HOGDescriptor()

#open the index file
indexFile = open(args["hindex"], 'w')
count = 1
for imagePath in glob.glob(args["dataset"] + "/*.jpg"):
    UID = imagePath[imagePath.rfind('/') + 1:]
    image = cv2.imread(imagePath)

    #describe the image
    features = hogd.describe(image)
    print "calculating..." + str(count) + "th image..."
    #output features to index file
    features = [str(f) for f in features]
    indexFile.write("%s, %s\n" % (UID, ",".join(features)))
    count += 1

#close index file
indexFile.close()
