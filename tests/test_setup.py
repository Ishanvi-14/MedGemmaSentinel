"""Test to verify pytest and Hypothesis setup."""

import pytest
from hypothesis import given, strategies as st


def test_pytest_works():
    """Verify pytest is working."""
    assert True


@pytest.mark.property
@given(st.integers())
def test_hypothesis_works(x):
    """Verify Hypothesis is configured correctly."""
    # This should run 100 times based on conftest.py settings
    assert isinstance(x, int)


def test_fixtures_exist():
    """Verify test fixtures directory exists."""
    import os
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    assert os.path.exists(fixtures_dir)
    
    # Check that all expected fixture files exist
    expected_files = [
        "sample_lab_report.txt",
        "sample_fhir.json",
        "sample_synthea.json",
        "sample_lab_results.csv",
        "sample_clinical_notes.txt"
    ]
    
    for filename in expected_files:
        filepath = os.path.join(fixtures_dir, filename)
        assert os.path.exists(filepath), f"Missing fixture: {filename}"
