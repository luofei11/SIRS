from colordescriptor import ColorDescriptor
from searcher import Searcher
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required = True, help = "Path to index file")
ap.add_argument("-q", "--query", required = True, help = "Path to query image")
ap.add_argument("-r", "--result_path", required = True, help = "Path to where result is stored")

args = vars(ap.parse_args())

cd = ColorDescriptor((8, 12, 3))
query = cv2.imread(args["query"])
feature = cd.describe(query)

searcher = Searcher(args["index"])
result = searcher.search(feature)

cv2.imshow("Query Image", query)

for (score, UID) in result:
    candidate = cv2.imread(args["result_path"] + "/" + UID)
    #cv2.imshow("Result", candidate)
    cv2.imwrite("results/" + UID, candidate)
    #cv2.waitKey(0)
