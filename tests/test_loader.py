"""Tests for the data loader module."""

import pytest
from pydantic import ValidationError
from irbstudio.data.loader import load_config


def test_load_config_success():
    """Tests that a valid config file is loaded correctly."""
    config = load_config("tests/data/valid_config.yaml")
    assert len(config.scenarios) == 1
    assert config.scenarios[0].name == "Baseline"
    assert config.scenarios[0].pd_auc == 0.75


def test_load_config_missing_scenarios_raises_error():
    """Tests that a config missing the required 'scenarios' field raises a validation error."""
    with pytest.raises(ValidationError):
        load_config("tests/data/invalid_config_missing_scenarios.yaml")


def test_load_config_bad_auc_raises_error():
    """Tests that a config with an out-of-bounds AUC value raises a validation error."""
    with pytest.raises(ValidationError):
        load_config("tests/data/invalid_config_bad_auc.yaml")


def test_load_config_file_not_found_raises_error():
    """Tests that trying to load a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("tests/data/non_existent_file.yaml")
