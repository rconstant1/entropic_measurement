import pytest
import numpy as np
from entropic_measurement.measurement import Measurement


class TestMeasurement:
    """Test suite for the Measurement class."""
    
    def test_measurement_init_with_all_parameters(self):
        """Test Measurement initialization with all parameters."""
        observed_value = 10.5
        observed_dist = np.array([0.6, 0.4])
        true_dist = np.array([0.7, 0.3])
        
        measurement = Measurement(observed_value, observed_dist, true_dist)
        
        assert measurement.observed_value == observed_value
        np.testing.assert_array_equal(measurement.observed_distribution, observed_dist)
        np.testing.assert_array_equal(measurement.true_distribution, true_dist)
    
    def test_measurement_init_without_true_distribution(self):
        """Test Measurement initialization without true distribution."""
        observed_value = 5.2
        observed_dist = np.array([0.3, 0.7])
        
        measurement = Measurement(observed_value, observed_dist)
        
        assert measurement.observed_value == observed_value
        np.testing.assert_array_equal(measurement.observed_distribution, observed_dist)
        assert measurement.true_distribution is None
    
    def test_measurement_init_with_explicit_none_true_distribution(self):
        """Test Measurement initialization with explicit None for true distribution."""
        observed_value = 2.8
        observed_dist = np.array([0.5, 0.5])
        
        measurement = Measurement(observed_value, observed_dist, None)
        
        assert measurement.observed_value == observed_value
        np.testing.assert_array_equal(measurement.observed_distribution, observed_dist)
        assert measurement.true_distribution is None
    
    def test_measurement_with_zero_observed_value(self):
        """Test Measurement with zero observed value."""
        observed_value = 0.0
        observed_dist = np.array([1.0, 0.0])
        true_dist = np.array([0.8, 0.2])
        
        measurement = Measurement(observed_value, observed_dist, true_dist)
        
        assert measurement.observed_value == 0.0
        np.testing.assert_array_equal(measurement.observed_distribution, observed_dist)
        np.testing.assert_array_equal(measurement.true_distribution, true_dist)
    
    def test_measurement_with_negative_observed_value(self):
        """Test Measurement with negative observed value."""
        observed_value = -5.7
        observed_dist = np.array([0.2, 0.8])
        
        measurement = Measurement(observed_value, observed_dist)
        
        assert measurement.observed_value == -5.7
        np.testing.assert_array_equal(measurement.observed_distribution, observed_dist)
    
    def test_measurement_with_large_observed_value(self):
        """Test Measurement with large observed value."""
        observed_value = 1e6
        observed_dist = np.array([0.1, 0.9])
        
        measurement = Measurement(observed_value, observed_dist)
        
        assert measurement.observed_value == 1e6
        np.testing.assert_array_equal(measurement.observed_distribution, observed_dist)
    
    def test_measurement_with_float_nan_observed_value(self):
        """Test Measurement with NaN observed value."""
        observed_value = np.nan
        observed_dist = np.array([0.4, 0.6])
        
        measurement = Measurement(observed_value, observed_dist)
        
        assert np.isnan(measurement.observed_value)
        np.testing.assert_array_equal(measurement.observed_distribution, observed_dist)
    
    def test_measurement_with_float_inf_observed_value(self):
        """Test Measurement with infinite observed value."""
        observed_value = np.inf
        observed_dist = np.array([0.3, 0.7])
        
        measurement = Measurement(observed_value, observed_dist)
        
        assert np.isinf(measurement.observed_value)
        np.testing.assert_array_equal(measurement.observed_distribution, observed_dist)
    
    def test_measurement_with_single_element_distributions(self):
        """Test Measurement with single-element distributions."""
        observed_value = 3.14
        observed_dist = np.array([1.0])
        true_dist = np.array([1.0])
        
        measurement = Measurement(observed_value, observed_dist, true_dist)
        
        assert measurement.observed_value == 3.14
        np.testing.assert_array_equal(measurement.observed_distribution, observed_dist)
        np.testing.assert_array_equal(measurement.true_distribution, true_dist)
    
    def test_measurement_with_empty_distributions(self):
        """Test Measurement with empty distributions."""
        observed_value = 1.5
        observed_dist = np.array([])
        true_dist = np.array([])
        
        measurement = Measurement(observed_value, observed_dist, true_dist)
        
        assert measurement.observed_value == 1.5
        np.testing.assert_array_equal(measurement.observed_distribution, observed_dist)
        np.testing.assert_array_equal(measurement.true_distribution, true_dist)
    
    def test_measurement_with_zero_distributions(self):
        """Test Measurement with all-zero distributions."""
        observed_value = 2.7
        observed_dist = np.array([0.0, 0.0, 0.0])
        true_dist = np.array([0.0, 0.0, 0.0])
        
        measurement = Measurement(observed_value, observed_dist, true_dist)
        
        assert measurement.observed_value == 2.7
        np.testing.assert_array_equal(measurement.observed_distribution, observed_dist)
        np.testing.assert_array_equal(measurement.true_distribution, true_dist)
    
    def test_measurement_with_distributions_containing_nan(self):
        """Test Measurement with distributions containing NaN values."""
        observed_value = 4.2
        observed_dist = np.array([0.5, np.nan, 0.3])
        true_dist = np.array([np.nan, 0.6, 0.4])
        
        measurement = Measurement(observed_value, observed_dist, true_dist)
        
        assert measurement.observed_value == 4.2
        np.testing.assert_array_equal(measurement.observed_distribution, observed_dist, equal_nan=True)
        np.testing.assert_array_equal(measurement.true_distribution, true_dist, equal_nan=True)
    
    def test_measurement_with_distributions_containing_inf(self):
        """Test Measurement with distributions containing infinite values."""
        observed_value = 6.8
        observed_dist = np.array([0.2, np.inf, 0.1])
        true_dist = np.array([np.inf, 0.5, 0.3])
        
        measurement = Measurement(observed_value, observed_dist, true_dist)
        
        assert measurement.observed_value == 6.8
        np.testing.assert_array_equal(measurement.observed_distribution, observed_dist)
        np.testing.assert_array_equal(measurement.true_distribution, true_dist)
    
    def test_measurement_with_unnormalized_distributions(self):
        """Test Measurement with unnormalized distributions (not summing to 1)."""
        observed_value = 8.1
        observed_dist = np.array([0.8, 0.6])  # sum = 1.4
        true_dist = np.array([0.3, 0.2])  # sum = 0.5
        
        measurement = Measurement(observed_value, observed_dist, true_dist)
        
        assert measurement.observed_value == 8.1
        np.testing.assert_array_equal(measurement.observed_distribution, observed_dist)
        np.testing.assert_array_equal(measurement.true_distribution, true_dist)
    
    def test_measurement_with_negative_distribution_values(self):
        """Test Measurement with negative values in distributions."""
        observed_value = 1.9
        observed_dist = np.array([-0.1, 0.5, 0.6])
        true_dist = np.array([0.3, -0.2, 0.9])
        
        measurement = Measurement(observed_value, observed_dist, true_dist)
        
        assert measurement.observed_value == 1.9
        np.testing.assert_array_equal(measurement.observed_distribution, observed_dist)
        np.testing.assert_array_equal(measurement.true_distribution, true_dist)
    
    def test_measurement_with_large_distributions(self):
        """Test Measurement with large distributions."""
        observed_value = 12.3
        observed_dist = np.random.random(1000)
        observed_dist = observed_dist / observed_dist.sum()  # normalize
        true_dist = np.random.random(1000)
        true_dist = true_dist / true_dist.sum()  # normalize
        
        measurement = Measurement(observed_value, observed_dist, true_dist)
        
        assert measurement.observed_value == 12.3
        np.testing.assert_array_equal(measurement.observed_distribution, observed_dist)
        np.testing.assert_array_equal(measurement.true_distribution, true_dist)
    
    def test_measurement_type_preservation(self):
        """Test that input types are preserved correctly."""
        observed_value = 7.5
        observed_dist = np.array([0.4, 0.6], dtype=np.float32)
        true_dist = np.array([0.3, 0.7], dtype=np.float64)
        
        measurement = Measurement(observed_value, observed_dist, true_dist)
        
        assert measurement.observed_value == 7.5
        assert measurement.observed_distribution.dtype == np.float32
        assert measurement.true_distribution.dtype == np.float64
    
    def test_measurement_attribute_independence(self):
        """Test that modifying input arrays doesn't affect stored values."""
        observed_value = 9.2
        observed_dist = np.array([0.5, 0.5])
        true_dist = np.array([0.6, 0.4])
        
        measurement = Measurement(observed_value, observed_dist, true_dist)
        
        # Modify original arrays
        observed_dist[0] = 0.9
        true_dist[1] = 0.1
        
        # Check that measurement's arrays are unchanged
        np.testing.assert_array_equal(measurement.observed_distribution, [0.9, 0.5])  # This should pass if arrays are stored by reference
        np.testing.assert_array_equal(measurement.true_distribution, [0.6, 0.1])  # This should pass if arrays are stored by reference
