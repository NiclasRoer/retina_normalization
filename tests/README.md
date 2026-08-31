# Tests for Retina Normalization

This directory contains the test suite for the retina_normalization package.

## Structure

- `test_dataset_functions.py` - Tests for dataset loading and data analysis
- `test_experiment_setup.py` - Tests for corruption functions and experiment utilities
- `test_model_functions.py` - Tests for model loading, adaptation, and forward passes
- `test_measurement_functions.py` - Tests for FLOP measurement utilities
- `test_retina_mechanisms.py` - Tests for retina preprocessing blocks
- `test_utility_functions.py` - Tests for argument parsing and utilities
- `conftest.py` - Pytest configuration and shared fixtures

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_dataset_functions.py
```

### Run specific test class
```bash
pytest tests/test_dataset_functions.py::TestDataAnalyst
```

### Run specific test
```bash
pytest tests/test_dataset_functions.py::TestDataAnalyst::test_infer_input_spec_4d_tensor
```

### Run with verbose output
```bash
pytest -v
```

### Run with coverage report
```bash
pytest --cov=packages --cov-report=html
```

### Run only fast tests (skip slow tests)
```bash
pytest -m "not slow"
```

### Run only GPU tests
```bash
pytest -m gpu
```

## Test Coverage

The test suite covers:

- **Dataset Functions**
  - Data loader construction
  - Dataset readers (LocalMNIST, ExampleDataset, CustomDataset)
  - Input specification inference
  - Number of classes inference

- **Experiment Setup**
  - Corruption functions (gaussian_noise, gaussian_blur, brightness_shift, contrast_shift)
  - Corruption dictionary and batch handling

- **Model Functions**
  - Model loading and discovery
  - Model architecture adaptation
  - Custom model implementation
  - Forward passes with various batch sizes
  - Gradient flow

- **Measurement Functions**
  - FLOP counting and measurement
  - Model state preservation during measurement

- **Retina Mechanisms**
  - RetinaPreprocessingBlock initialization
  - Module sequencing and device handling
  - Train/eval mode switching

- **Utility Functions**
  - Command-line argument parsing
  - Multiple argument handling

## Adding New Tests

When adding new functionality to the packages, please add corresponding tests following these guidelines:

1. Create test methods named `test_*` in test classes named `Test*`
2. Use descriptive docstrings explaining what is being tested
3. Use pytest fixtures from `conftest.py` when appropriate
4. Group related tests in test classes
5. Use parametrized tests for testing multiple inputs
6. Mark slow tests with `@pytest.mark.slow`
7. Mark GPU-dependent tests with `@pytest.mark.gpu`

Example:
```python
class TestNewFeature:
    """Tests for new feature."""
    
    def test_basic_functionality(self):
        """Test that basic functionality works."""
        # Test code here
        pass
    
    @pytest.mark.slow
    def test_expensive_operation(self):
        """Test expensive operation."""
        # Slow test code here
        pass
```

## Dependencies

Test dependencies are specified in `pyproject.toml`:
- pytest >= 7.0
- pytest-cov >= 4.0

Install with:
```bash
pip install -e ".[test]"
```
