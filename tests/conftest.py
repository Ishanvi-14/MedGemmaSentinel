"""Pytest configuration and fixtures for Sentinel tests."""

from hypothesis import settings, Verbosity

# Configure Hypothesis for 100 iterations per property test
settings.register_profile("sentinel", max_examples=100, verbosity=Verbosity.verbose)
settings.load_profile("sentinel")
