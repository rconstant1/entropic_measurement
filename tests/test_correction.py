import numpy as np
import pytest
from entropic_measurement.correction import EntropicBiasCorrector
from entropic_measurement.measurement import Measurement

class DummyEstimator:
    @staticmethod
    def kullback_leibler(p, q):
        p, q = np.asarray(p), np.asarray(q)
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-12
        return np.sum(np.where(p != 0, p * np.log((p + epsilon)/(q + epsilon)), 0.0))

@pytest.fixture(autouse=True)
def patch_entropy(monkeypatch):
    import entropic_measurement.correction
    entropic_measurement.correction.EntropyEstimator = DummyEstimator

@pytest.mark.parametrize("observed, true, obs_val, expected_corr", [
    # Nominal (normal distribution)
    ([0.6, 0.4], [0.5, 0.5], 1.0, np.exp(DummyEstimator.kullback_leibler([0.5, 0.5],[0.6, 0.4]))),
    # Zero distribution
    ([1.0, 0.0], [1.0, 0.0], 5.0, 5.0),
    # NaN case (should propagate NaN if present)
    ([np.nan, np.nan], [0.5, 0.5], 1.0, np.nan),
    # Floats
    ([0.75, 0.25], [0.7, 0.3], 2.1, 2.1 * np.exp(DummyEstimator.kullback_leibler([0.7, 0.3],[0.75, 0.25]))),
])
def test_correct_nominal_zero_nan_float(observed, true, obs_val, expected_corr):
    m = Measurement(observed_value=obs_val, observed_distribution=observed, true_distribution=true)
    corrector = EntropicBiasCorrector()
    result = corrector.correct(m)
    if np.isnan(expected_corr):
        assert np.isnan(result['corrected_value'])
    else:
        assert np.allclose(result['corrected_value'], expected_corr, equal_nan=True)

def test_correct_no_true_distribution():
    m = Measurement(observed_value=1.0, observed_distribution=[0.5, 0.5], true_distribution=None)
    corrector = EntropicBiasCorrector()
    result = corrector.correct(m)
    assert result['corrected_value'] is None
    assert result['kl_divergence'] is None
    assert result['entropy_cost'] is None
@pytest.mark.parametrize("obs,tru,obs_val", [
    ([0.0], [0.0], 0.0),
    ([], [], 0.0),
])
def test_correct_degenerate_arrays(obs, tru, obs_val):
    m = Measurement(observed_value=obs_val, observed_distribution=obs, true_distribution=tru)
    corrector = EntropicBiasCorrector()
    result = corrector.correct(m)
    # Degenerate arrays: expect KL ~ 0, correction should not fail
    assert np.isfinite(result['corrected_value']) or result['corrected_value'] is None
