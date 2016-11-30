import numpy as np
import csv
from similarity import ImageSimilarity

class Searcher:

    def __init__(self, indexPath):
        #path to index file
        self.indexPath = indexPath

    def search(self, query, limit = 10):
        #initialize the results dictionary
        results = {}

        #open index file
        with open(self.indexPath) as f:
            csvReader = csv.reader(f)

            for row in csvReader:
                feature = [float(x) for x in row[1:]]
                sim = ImageSimilarity.calcChi2Distance(feature, query)

                results[row[0]] = sim
            #close file
            f.close()

        res = []
        for k in results:
            res.append((results[k], k))
        res.sort()
        return res[:limit]
