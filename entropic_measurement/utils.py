import numpy as np
from .measurement import Measurement
from .correction import EntropicBiasCorrector

def measure_and_correct(observed_value, observed_dist, true_dist=None, beta=1.0):
    measurement = Measurement(observed_value, np.array(observed_dist), np.array(true_dist) if true_dist is not None else None)
    corrector = EntropicBiasCorrector(beta)
    result = corrector.correct(measurement)
    return result
