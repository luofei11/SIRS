import numpy as np
import csv
from similarity import ImageSimilarity

class Searcher:

    def __init__(self, cindexPath, hindexPath):
        #path to index file
        self.cindexPath = cindexPath
        self.hindexPath = hindexPath

    def search(self, cquery, hquery, limit = 10):
        #initialize the results dictionary
        results = {}

        #open index file
        with open(self.cindexPath) as f:
            csvReader = csv.reader(f)

            for row in csvReader:
                feature = [float(x) for x in row[1:]]
                sim = ImageSimilarity.calcChi2Distance(feature, cquery)

                results[row[0]] = sim
            #close file
            f.close()

        with open(self.hindexPath) as f:
            csvReader = csv.reader(f)

            for row in csvReader:
                feature = [float(x) for x in row[1:]]
                sim = ImageSimilarity.calcOrdinalDistance(feature, hquery)

                results[row[0]] += sim
            #close file
            f.close()


        res = []
        for k in results:
            res.append((results[k], k))
        res.sort()
        return res[:limit]
