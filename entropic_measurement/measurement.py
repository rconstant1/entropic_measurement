import numpy as np
from typing import Optional

class Measurement:
    """Generic class for storing a measurement, its observed and (optionally) true distributions."""
    def __init__(self, observed_value: float, observed_distribution: np.ndarray, true_distribution: Optional[np.ndarray] = None):
        self.observed_value = observed_value
        self.observed_distribution = observed_distribution
        self.true_distribution = true_distribution
