import numpy as np

class ImageSimilarity:

    @staticmethod
    def calcChi2Distance(v1, v2, eps = 1e-10):
        #return chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(v1, v2)])

        return d
