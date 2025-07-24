import pytest

# Adjust the import below to match the actual location in your package
from entropic_measurement.entropy import shannon_entropy_tensor

@pytest.mark.parametrize("backend", ["torch", "tf"])
def test_shannon_entropy_tensor_nominal(backend):
    if backend == "torch":
        pytest.importorskip("torch")
        import torch
        x = torch.tensor([0.5, 0.5], dtype=torch.float32)
        result = shannon_entropy_tensor(x)
        assert abs(result.item() - 1.0) < 1e-4  # Entropy([0.5,0.5]) == 1.0

    if backend == "tf":
        pytest.importorskip("tensorflow")
        import tensorflow as tf
        x = tf.constant([0.5, 0.5], dtype=tf.float32)
        result = shannon_entropy_tensor(x)
        assert abs(result.numpy() - 1.0) < 1e-4

def test_shannon_entropy_tensor_zero_case():
    import torch
    x = torch.tensor([1.0, 0.0, 0.0])
    val = shannon_entropy_tensor(x)
    assert val.item() == 0

def test_shannon_entropy_tensor_nan_guard():
    import torch
    x = torch.tensor([float('nan'), 1.0])
    # Should not fail but result will be nan
    result = shannon_entropy_tensor(x)
    assert result != result  # nan != nan

def test_shannon_entropy_tensor_wrong_type():
    # Passing unsupported type → must raise TypeError
    with pytest.raises(TypeError):
        shannon_entropy_tensor([0.5, 0.5])  # List, not tensor

def test_shannon_entropy_tensor_backend_unavailable(monkeypatch):
    # Simulate both torch and tf unavailable
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    # The actual test will depend on your fallback behavior

