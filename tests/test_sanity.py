import pytest

def test_sanity():
    """
    A basic sanity check test to ensure the testing framework is set up correctly.
    """
    assert 1 + 1 == 2

def test_pytest_installed():
    """
    Checks that pytest is installed and importable.
    """
    try:
        import pytest
    except ImportError:
        pytest.fail("Pytest is not installed or accessible in the environment.")
