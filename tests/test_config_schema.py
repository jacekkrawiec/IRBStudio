import pytest
from pydantic import ValidationError
from irbstudio.config.schema import Config, Scenario


def test_config_requires_at_least_one_scenario():
    with pytest.raises(ValidationError):
        Config(scenarios=[])


def test_config_parses_single_scenario():
    c = Config(
        scenarios=[
            Scenario(
                name="base", pd_auc=0.75, portfolio_default_rate=0.02, lgd=0.45
            )
        ]
    )
    assert len(c.scenarios) == 1
    assert c.scenarios[0].pd_auc == 0.75
    assert c.scenarios[0].portfolio_default_rate == 0.02
    assert c.scenarios[0].lgd == 0.45


def test_scenario_validation_rules():
    # AUC must be between 0.5 and 1.0
    with pytest.raises(ValidationError):
        Scenario(name="bad_auc", pd_auc=1.1, portfolio_default_rate=0.02, lgd=0.45)
    with pytest.raises(ValidationError):
        Scenario(name="bad_auc", pd_auc=0.4, portfolio_default_rate=0.02, lgd=0.45)

    # PDR must be between 0 and 1
    with pytest.raises(ValidationError):
        Scenario(name="bad_pdr", pd_auc=0.75, portfolio_default_rate=-0.1, lgd=0.45)

    # LGD must be between 0 and 1
    with pytest.raises(ValidationError):
        Scenario(name="bad_lgd", pd_auc=0.75, portfolio_default_rate=0.02, lgd=1.5)
