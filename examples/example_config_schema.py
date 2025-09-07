"""
Example: Using the IRBStudio config schema directly (Feature 1.2)
"""
from irbstudio.config.schema import Config, Scenario

# Create a config object directly in Python
config = Config(
    scenarios=[
        Scenario(name="Baseline", pd_auc=0.75, description="A sample baseline scenario")
    ]
)

print("Config object created and validated:")
print(config)
