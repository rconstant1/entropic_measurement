import pytest
import numpy as np
from entropic_measurement import utils

def test_utils_nominal():
    # Remplace par un véritable appel à une fonction publique de utils.py
    assert utils.some_function(1, 2) == 3

def test_utils_zero():
    assert utils.some_function(0, 0) == 0

def test_utils_nan():
    with pytest.raises(ValueError):
        utils.some_function(np.nan, 2)

def test_utils_float_extreme():
    assert np.isfinite(utils.some_function(1e100, 1e-100))

def test_utils_array_degenerate():
    with pytest.raises(Exception):
        utils.some_function([])
