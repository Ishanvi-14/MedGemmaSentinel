# Sentinel Test Suite

This directory contains the test suite for the Sentinel competition enhancements.

## Structure

```
tests/
├── __init__.py
├── conftest.py              # Pytest configuration with Hypothesis settings
├── test_setup.py            # Setup verification tests
├── fixtures/                # Test data files
│   ├── sample_lab_report.txt
│   ├── sample_fhir.json
│   ├── sample_synthea.json
│   ├── sample_lab_results.csv
│   ├── sample_clinical_notes.txt
│   └── README.md
└── README.md               # This file
```

## Running Tests

### Run all tests
```bash
python -m pytest
```

### Run with verbose output
```bash
python -m pytest -v
```

### Run with coverage report
```bash
python -m pytest --cov=src --cov-report=html
```

### Run only property tests
```bash
python -m pytest -m property
```

### Run only unit tests
```bash
python -m pytest -m unit
```

## Test Configuration

### Hypothesis Settings
- Profile: `sentinel`
- Max examples per property test: 100
- Verbosity: Verbose

### Coverage Settings
- Source directory: `src/`
- Reports: HTML and terminal
- Branch coverage: Enabled

## Test Markers

- `@pytest.mark.property` - Property-based tests using Hypothesis
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow-running tests

## Writing Tests

### Property-Based Tests
```python
from hypothesis import given, strategies as st
import pytest

@pytest.mark.property
@given(st.integers())
def test_my_property(value):
    # Test will run 100 times with different values
    assert my_function(value) >= 0
```

### Unit Tests
```python
import pytest

@pytest.mark.unit
def test_my_function():
    result = my_function(42)
    assert result == expected_value
```

## Fixtures

Test fixtures are located in `tests/fixtures/` and include sample clinical data in multiple formats:
- PDF (text representation)
- FHIR JSON
- Synthea JSON
- CSV
- Plain text clinical notes

All fixtures contain consistent biomarker data for patient P12345 to enable cross-format validation.
