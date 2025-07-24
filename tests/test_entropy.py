import pytest
import numpy as np
from entropy import calculate_entropy, shannon_entropy


class TestCalculateEntropy:
    """Test cases for calculate_entropy function."""
    
    def test_nominal_case_uniform_distribution(self):
        """Test entropy calculation for uniform distribution."""
        # Uniform distribution should have maximum entropy
        data = np.array([1, 1, 1, 1])
        result = calculate_entropy(data)
        expected = np.log2(4)  # log2(n) for uniform distribution
        assert np.isclose(result, expected)
    
    def test_nominal_case_non_uniform_distribution(self):
        """Test entropy calculation for non-uniform distribution."""
        data = np.array([1, 2, 1, 1])
        result = calculate_entropy(data)
        # Expected: -((3/5)*log2(3/5) + (2/5)*log2(2/5))
        p1, p2 = 3/5, 2/5
        expected = -(p1 * np.log2(p1) + p2 * np.log2(p2))
        assert np.isclose(result, expected)
    
    def test_nominal_case_single_element(self):
        """Test entropy calculation for single element (zero entropy)."""
        data = np.array([5, 5, 5, 5])
        result = calculate_entropy(data)
        assert np.isclose(result, 0.0)
    
    def test_zero_values_mixed(self):
        """Test entropy calculation with zero values mixed with non-zero."""
        data = np.array([0, 1, 0, 1, 2])
        result = calculate_entropy(data)
        # Only non-zero values should be considered: [1, 1, 2]
        # Probabilities: 2/3 for value 1, 1/3 for value 2
        p1, p2 = 2/3, 1/3
        expected = -(p1 * np.log2(p1) + p2 * np.log2(p2))
        assert np.isclose(result, expected)
    
    def test_all_zeros(self):
        """Test entropy calculation with all zero values."""
        data = np.array([0, 0, 0, 0])
        result = calculate_entropy(data)
        # Should return 0 or handle gracefully
        assert result == 0.0 or np.isnan(result)
    
    def test_nan_values_in_array(self):
        """Test entropy calculation with NaN values in input."""
        data = np.array([1, np.nan, 2, 1])
        with pytest.raises((ValueError, TypeError)) or pytest.warns(RuntimeWarning):
            calculate_entropy(data)
    
    def test_all_nan_values(self):
        """Test entropy calculation with all NaN values."""
        data = np.array([np.nan, np.nan, np.nan])
        with pytest.raises((ValueError, TypeError)) or pytest.warns(RuntimeWarning):
            calculate_entropy(data)
    
    def test_extreme_float_values_large(self):
        """Test entropy calculation with extremely large float values."""
        data = np.array([1e308, 1e308, 2e308])
        result = calculate_entropy(data)
        # Should handle large values gracefully
        assert not np.isnan(result)
        assert result >= 0
    
    def test_extreme_float_values_small(self):
        """Test entropy calculation with extremely small float values."""
        data = np.array([1e-308, 1e-308, 2e-308])
        result = calculate_entropy(data)
        # Should handle small values gracefully
        assert not np.isnan(result)
        assert result >= 0
    
    def test_infinite_values(self):
        """Test entropy calculation with infinite values."""
        data = np.array([np.inf, 1, 2, np.inf])
        with pytest.raises((ValueError, TypeError)) or pytest.warns(RuntimeWarning):
            calculate_entropy(data)
    
    def test_negative_infinite_values(self):
        """Test entropy calculation with negative infinite values."""
        data = np.array([-np.inf, 1, 2])
        with pytest.raises((ValueError, TypeError)) or pytest.warns(RuntimeWarning):
            calculate_entropy(data)


class TestDegenerateArrays:
    """Test cases for degenerate array inputs."""
    
    def test_empty_array(self):
        """Test entropy calculation with empty array."""
        data = np.array([])
        with pytest.raises((ValueError, IndexError)):
            calculate_entropy(data)
    
    def test_single_element_array(self):
        """Test entropy calculation with single element array."""
        data = np.array([42])
        result = calculate_entropy(data)
        assert np.isclose(result, 0.0)
    
    def test_two_element_array_same(self):
        """Test entropy calculation with two identical elements."""
        data = np.array([5, 5])
        result = calculate_entropy(data)
        assert np.isclose(result, 0.0)
    
    def test_two_element_array_different(self):
        """Test entropy calculation with two different elements."""
        data = np.array([1, 2])
        result = calculate_entropy(data)
        expected = 1.0  # log2(2) = 1 for two equally likely outcomes
        assert np.isclose(result, expected)
    
    def test_multidimensional_array(self):
        """Test entropy calculation with multidimensional array."""
        data = np.array([[1, 2], [1, 3]])
        # Should either flatten or raise appropriate error
        try:
            result = calculate_entropy(data)
            assert not np.isnan(result)
            assert result >= 0
        except (ValueError, TypeError):
            # Acceptable if function doesn't support multidimensional
            pass
    
    def test_boolean_array(self):
        """Test entropy calculation with boolean array."""
        data = np.array([True, False, True, True])
        result = calculate_entropy(data)
        # True appears 3 times, False appears 1 time
        p1, p2 = 3/4, 1/4
        expected = -(p1 * np.log2(p1) + p2 * np.log2(p2))
        assert np.isclose(result, expected)


class TestShannonEntropy:
    """Test cases for shannon_entropy function if it exists."""
    
    def test_shannon_entropy_nominal(self):
        """Test Shannon entropy calculation for nominal case."""
        try:
            probabilities = np.array([0.5, 0.3, 0.2])
            result = shannon_entropy(probabilities)
            expected = -(0.5 * np.log2(0.5) + 0.3 * np.log2(0.3) + 0.2 * np.log2(0.2))
            assert np.isclose(result, expected)
        except (NameError, ImportError):
            pytest.skip("shannon_entropy function not available")
    
    def test_shannon_entropy_uniform(self):
        """Test Shannon entropy for uniform distribution."""
        try:
            probabilities = np.array([0.25, 0.25, 0.25, 0.25])
            result = shannon_entropy(probabilities)
            expected = 2.0  # log2(4)
            assert np.isclose(result, expected)
        except (NameError, ImportError):
            pytest.skip("shannon_entropy function not available")
    
    def test_shannon_entropy_invalid_probabilities(self):
        """Test Shannon entropy with invalid probability values."""
        try:
            # Probabilities that don't sum to 1
            probabilities = np.array([0.3, 0.3, 0.3])
            with pytest.raises(ValueError):
                shannon_entropy(probabilities)
        except (NameError, ImportError):
            pytest.skip("shannon_entropy function not available")


class TestEdgeCases:
    """Additional edge case tests."""
    
    def test_very_large_array(self):
        """Test entropy calculation with very large array."""
        data = np.random.randint(1, 100, size=10000)
        result = calculate_entropy(data)
        assert not np.isnan(result)
        assert result >= 0
    
    def test_negative_values(self):
        """Test entropy calculation with negative values."""
        data = np.array([-1, -2, -1, -3])
        result = calculate_entropy(data)
        # Should treat negative values as distinct categories
        assert not np.isnan(result)
        assert result >= 0
    
    def test_mixed_positive_negative(self):
        """Test entropy calculation with mixed positive and negative values."""
        data = np.array([-1, 1, -1, 2, 1])
        result = calculate_entropy(data)
        assert not np.isnan(result)
        assert result >= 0
    
    def test_floating_point_precision(self):
        """Test entropy calculation with floating point precision issues."""
        data = np.array([0.1 + 0.2, 0.3, 0.1 + 0.2])  # 0.3 vs 0.30000000000000004
        result = calculate_entropy(data)
        # Should handle floating point precision gracefully
        assert not np.isnan(result)
        assert result >= 0


if __name__ == "__main__":
    pytest.main([__file__])
