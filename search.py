from colordescriptor import ColorDescriptor
from hogdescriptor import HOGDescriptor
from searcher import Searcher
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-ci", "--cindex", required = True, help = "Path to index file")
ap.add_argument("-hi", "--hindex", required = True, help = "Path to index file")
ap.add_argument("-q", "--query", required = True, help = "Path to query image")
ap.add_argument("-r", "--result_path", required = True, help = "Path to where result is stored")

args = vars(ap.parse_args())

cd = ColorDescriptor((8, 12, 3))
hogd = HOGDescriptor()
query = cv2.imread(args["query"])
feature1 = cd.describe(query)
feature2 = hogd.describe(query)
searcher = Searcher(args["cindex"], args["hindex"])
result = searcher.search(feature1, feature2)

cv2.imshow("Query Image", query)

for (score, UID) in result:
    candidate = cv2.imread(args["result_path"] + "/" + UID)
    #cv2.imshow("Result", candidate)
    cv2.imwrite("results/" + UID, candidate)
    #cv2.waitKey(0)
