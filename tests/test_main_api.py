"""
Tests for main API (run_analysis and run_scenario_comparison).

Priority 1: These tests are placeholders for the high-level API functions
that will be implemented in irbstudio.main module.
"""

import pytest

# Note: These tests are placeholders since run_analysis and run_scenario_comparison
# are not yet implemented in the main module. They are marked as skipped.


@pytest.mark.skip(reason="run_analysis() not yet implemented in main module")
def test_run_analysis_basic():
    """
    Test run_analysis() with minimal parameters.
    
    This test verifies that the high-level run_analysis() function can:
    - Load a portfolio from a CSV file
    - Load configuration from a YAML file
    - Run the complete analysis workflow
    - Return results in the expected format
    """
    pass


@pytest.mark.skip(reason="run_scenario_comparison() not yet implemented")  
def test_run_scenario_comparison_basic():
    """
    Test run_scenario_comparison() with two configurations.
    
    This test verifies that scenario comparison can:
    - Run two separate analyses with different configurations
    - Calculate capital delta between scenarios
    - Return comparison results
    """
    pass
