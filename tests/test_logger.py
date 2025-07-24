import pytest
import numpy as np
from entropic_measurement.logger import EntropicLogger

def test_logger_nominal(tmp_path):
    logger = EntropicLogger()
    result = {'observed_value': 1.0, 'corrected_value': 1.1, 'kl_divergence': 0.1, 'entropy_cost': 0.1}
    logger.record(result)
    log_file = tmp_path / "audit_log.csv"
    logger.export(str(log_file), format="csv")
    content = log_file.read_text()
    assert "observed_value" in content and "1.1" in content

def test_logger_zero(tmp_path):
    logger = EntropicLogger()
    result = {'observed_value': 0, 'corrected_value': 0, 'kl_divergence': 0, 'entropy_cost': 0}
    logger.record(result)
    log_file = tmp_path / "audit_zero.csv"
    logger.export(str(log_file), format="csv")
    content = log_file.read_text()
    assert "0" in content

def test_logger_nan(tmp_path):
    logger = EntropicLogger()
    result = {'observed_value': np.nan, 'corrected_value': np.nan, 'kl_divergence': np.nan, 'entropy_cost': np.nan}
    logger.record(result)
    log_file = tmp_path / "audit_nan.csv"
    logger.export(str(log_file), format="csv")
    content = log_file.read_text()
    assert "nan" in content.lower()

def test_logger_float(tmp_path):
    logger = EntropicLogger()
    result = {'observed_value': 1e100, 'corrected_value': -1e-100, 'kl_divergence': 1e-50, 'entropy_cost': 1e50}
    logger.record(result)
    log_file = tmp_path / "audit_float.csv"
    logger.export(str(log_file), format="csv")
    content = log_file.read_text()
    assert "1e+100" in content or "-1e-100" in content

def test_logger_degenerate(tmp_path):
    logger = EntropicLogger()
    with pytest.raises(Exception):
        logger.record({})
