import numpy as np

class EntropyEstimator:
    @staticmethod
    def shannon_entropy(distribution: np.ndarray) -> float:
        p = distribution[distribution > 0]
        return -np.sum(p * np.log2(p))

    @staticmethod
    def kullback_leibler(P: np.ndarray, Q: np.ndarray) -> float:
        mask = (P > 0) & (Q > 0)
        return np.sum(P[mask] * np.log(P[mask] / Q[mask]))
