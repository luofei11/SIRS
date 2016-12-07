import numpy as np
import math
class ImageSimilarity:

    @staticmethod
    def calcChi2Distance(v1, v2, eps = 1e-10):
        #return chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(v1, v2)])

        return d

    @staticmethod
    def calcOrdinalDistance(v1, v2):
        dot = 0.0
        denom_a = 0.0
        denom_b = 0.0
        for i in range(min(len(v1), len(v2))):
            dot += v1[i] * v2[i];
            denom_a += v1[i] * v1[i];
            denom_b += v2[i] * v2[i];

        return dot / (math.sqrt(denom_a) * math.sqrt(denom_b));
