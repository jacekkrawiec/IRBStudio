import pytest
from pydantic import ValidationError
from irbstudio.config.schema import Config, Scenario


def test_config_requires_at_least_one_scenario():
    with pytest.raises(ValidationError):
        Config(scenarios=[])


def test_config_parses_single_scenario():
    c = Config(scenarios=[Scenario(name="base", pd_auc=0.75)])
    assert len(c.scenarios) == 1
    assert c.scenarios[0].pd_auc == 0.75
